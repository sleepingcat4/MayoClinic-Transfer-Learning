# Loading the model
model = geffnet.create_model('efficientnet_b2', pretrained=True)

# Freezing all the layers
for param in model.parameters():
  param.requires_grad = False

# Changing the Classifier
model.classifier = nn.Sequential(nn.Linear(1408,512),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(512,128),
                           nn.ReLU(),
                           nn.Dropout(p=0.4),
                           nn.Linear(128,len(classes)))

# Making the Classifier layer Trainable                           
for param in model.classifier.parameters():
  param.requires_grad = True

# Moving the model to device
model.to(device)

