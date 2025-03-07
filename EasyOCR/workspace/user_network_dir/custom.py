import torch
import torch.nn as nn
import time

class Model(nn.Module):

    def __init__(self, input_channel, output_channel, hidden_size, num_class):
        super(Model, self).__init__()
        """ FeatureExtraction """
        self.FeatureExtraction = VGG_FeatureExtractor(input_channel, output_channel)
        self.FeatureExtraction_output = output_channel
        self.AdaptiveAvgPool = nn.AdaptiveAvgPool2d((None, 1))

        """ Sequence modeling"""
        self.SequenceModeling = nn.Sequential(
            BidirectionalLSTM(self.FeatureExtraction_output, hidden_size, hidden_size),
            BidirectionalLSTM(hidden_size, hidden_size, hidden_size))
        self.SequenceModeling_output = hidden_size

        """ Prediction """
        self.Prediction = nn.Linear(self.SequenceModeling_output, num_class)

        # 모듈별 시간 저장을 위한 변수 초기화
        self.elapsed_times = {'vgg': 0.0, 'bilstm': 0.0, 'ctc': 0.0}

    def forward(self, input, text):
        if torch.cuda.is_available():
            # CUDA 이벤트 생성
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)

            # VGG 모듈 시간 측정
            start_event.record()
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
            visual_feature = visual_feature.squeeze(3)
            end_event.record()
            torch.cuda.synchronize()
            vgg_time = start_event.elapsed_time(end_event) / 1000.0  # 밀리초를 초로 변환
            self.elapsed_times['vgg'] = vgg_time

            # BiLSTM 모듈 시간 측정
            start_event.record()
            contextual_feature = self.SequenceModeling(visual_feature)
            end_event.record()
            torch.cuda.synchronize()
            bilstm_time = start_event.elapsed_time(end_event) / 1000.0
            self.elapsed_times['bilstm'] = bilstm_time

            # CTC 모듈 시간 측정
            start_event.record()
            prediction = self.Prediction(contextual_feature.contiguous())
            end_event.record()
            torch.cuda.synchronize()
            ctc_time = start_event.elapsed_time(end_event) / 1000.0
            self.elapsed_times['ctc'] = ctc_time
        else:
            # CPU 연산의 경우 time.perf_counter() 사용
            start_time = time.perf_counter()
            visual_feature = self.FeatureExtraction(input)
            visual_feature = self.AdaptiveAvgPool(visual_feature.permute(0, 3, 1, 2))
            visual_feature = visual_feature.squeeze(3)
            vgg_time = time.perf_counter() - start_time
            self.elapsed_times['vgg'] = vgg_time

            start_time = time.perf_counter()
            contextual_feature = self.SequenceModeling(visual_feature)
            bilstm_time = time.perf_counter() - start_time
            self.elapsed_times['bilstm'] = bilstm_time

            start_time = time.perf_counter()
            prediction = self.Prediction(contextual_feature.contiguous())
            ctc_time = time.perf_counter() - start_time
            self.elapsed_times['ctc'] = ctc_time

        return prediction

class BidirectionalLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, output_size):
        super(BidirectionalLSTM, self).__init__()
        self.rnn = nn.LSTM(input_size, hidden_size, bidirectional=True, batch_first=True)
        self.linear = nn.Linear(hidden_size * 2, output_size)

    def forward(self, input):
        """
        input : visual feature [batch_size x T x input_size]
        output : contextual feature [batch_size x T x output_size]
        """
        try: # multi gpu needs this
            self.rnn.flatten_parameters()
        except: # quantization doesn't work with this
            pass
        recurrent, _ = self.rnn(input)  # batch_size x T x input_size -> batch_size x T x (2*hidden_size)
        output = self.linear(recurrent)  # batch_size x T x output_size
        return output

class VGG_FeatureExtractor(nn.Module):

    def __init__(self, input_channel, output_channel=256):
        super(VGG_FeatureExtractor, self).__init__()
        self.output_channel = [int(output_channel / 8), int(output_channel / 4),
                               int(output_channel / 2), output_channel]
        self.ConvNet = nn.Sequential(
            nn.Conv2d(input_channel, self.output_channel[0], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[0], self.output_channel[1], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(self.output_channel[1], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.Conv2d(self.output_channel[2], self.output_channel[2], 3, 1, 1), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[2], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 3, 1, 1, bias=False),
            nn.BatchNorm2d(self.output_channel[3]), nn.ReLU(True),
            nn.MaxPool2d((2, 1), (2, 1)),
            nn.Conv2d(self.output_channel[3], self.output_channel[3], 2, 1, 0), nn.ReLU(True))

    def forward(self, input):
        return self.ConvNet(input)
