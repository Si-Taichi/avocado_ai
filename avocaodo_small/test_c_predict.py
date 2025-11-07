import test_prediction_c as predict

vis_path = "data/test/test_out_4.webp"

model_path = "models/best_model.pth"
classes = [str(i) for i in range(1, 11)]

label, conf = predict.predict(vis_path, model_path, classes)
print(f"Predicted day: {label}, confidence {conf:.2f}")
