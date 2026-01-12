def predict_video(video_path, model_path):
    model = DeepfakeModel()
    model.load_state_dict(torch.load(model_path))
    model.eval()
    # (Combine with preprocess logic to extract frames and get confidence scores)