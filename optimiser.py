criterion = nn.CrossEntropyLoss()
optimizer  = optim.Adam(model.classifier.parameters(),lr = 0.01)
scheduler = StepLR(optimizer, step_size=6, gamma=0.35)
