def test(img_path_list, true_class_list):
    from PIL import Image
    import torch
    from torchvision import models
    from torchvision import transforms

    class resnet(torch.nn.Module):
        def __init__(self, pretrained=False):
            super(resnet, self).__init__()
            if pretrained == True:
                model = models.resnet50(pretrained = True)
            else:
                model = models.resnet50(pretrained = False)

            self.slice1 = torch.nn.Sequential()
            self.slice1.add_module(str(1), model.conv1)
            self.slice1.add_module(str(2), model.bn1)
            self.slice1.add_module(str(3), model.relu)
            self.slice1.add_module(str(4), model.maxpool)
            self.slice1.add_module(str(5), model.layer1)
            self.slice1.add_module(str(6), model.layer2)
            self.slice1.add_module(str(7), model.layer3)
            
            model_other = models.resnet50(pretrained = False)

            self.slice2 = model_other.layer4
            self.avgpool = model_other.avgpool

            self.classifier = torch.nn.Sequential()
            self.classifier.add_module(str(1), model_other.fc)
            self.classifier.add_module(str(2), torch.nn.ReLU(inplace=True))
            self.classifier.add_module(str(3), torch.nn.Linear(1000, 2))

        def forward(self, x):
            x = self.slice1(x)
            x = self.slice2(x)
            x = self.avgpool(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
            return x

    # load the model
    model = resnet(pretrained=False)
    checkpoint = torch.load('./checkpoint_epoch_200.pth.tar')
    model.load_state_dict(checkpoint['state_dict'])
    model = model.cuda()
    model.eval()

    # prepare data
    transform = transforms.Compose([
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ])

    TP, TN, FP, FN = 0, 0, 0, 0
    for img_path, true_class in zip(img_path_list, true_class_list):
        img = Image.open(img_path)
        img = transform(img).unsqueeze(0)
        img = img.cuda()
        with torch.no_grad():
            output = model(img)
        pred_class = torch.argmax(output).item()

        if true_class == 1 and pred_class == 1:
            TP += 1
        elif true_class == 0 and pred_class == 0:
            TN += 1
        elif true_class == 0 and pred_class == 1:
            FP += 1
        elif true_class == 1 and pred_class == 0:
            FN += 1

    accuracy = (TP + TN) / (TP + TN + FP + FN)
    precision = (TP) / (TP + FP)
    recall = (TP) / (TP + FN)

    return accuracy, recall, precision