with torch.no_grad():
    # visualize predictions of the model
    # take a random picture from a dataset
    idx = np.random.randint(len(train_dataset))
    image, label = train_dataset[idx]
    # predict new bbox
    image = torch.unsqueeze(image, 0)
    outputs = model(image)
    outputs = outputs.flatten()
    # print(f'outputs shape: {outputs}')
    pred_bb = np.round(outputs[:-1], 4).tolist()  # <---- pred bb
    # print(f'pred bb shape: {pred_bb}')
    pred_start = (int(pred_bb[0]*crop), int(pred_bb[1]*crop))
    pred_end = (int(pred_bb[2]*crop), int(pred_bb[3]*crop))
    pred_id = outputs[-1].long().item()
    # need to permute tensor for visualization
    image = torch.squeeze(image, 0)
    image = np.array(image.permute(1, 2, 0))
    # bbox = first 4 digits
    bbox = np.round(label[:-1], 4).tolist()  # <---- true bb
    # class = last digit
    class_id = label[-1].long().item()
    # define 2 points for rectangle
    start_point = (int(bbox[0]*crop), int(bbox[1]*crop))
    end_point = (int(bbox[2]*crop), int(bbox[3]*crop))
    # define a color and thickness
    color = (0, 255, 0)
    color2 = (255, 0, 0)

    thickness = 2

    image = cv2.rectangle(image, start_point, end_point, color, thickness)
    image = cv2.rectangle(image, pred_start, pred_end, color2, thickness)

    # cv2.imshow('image', image)

    plt.figure(figsize=(5, 5))
    plt.imshow(np.array(image))
