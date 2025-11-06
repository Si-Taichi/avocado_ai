import test_predict as predict

vis_path = "path/to/visible.jpg"
nir_path = "path/to/nir.jpg"
model_path = "models/cnn4channel_avocado.pth"
classes = [str(i) for i in range(1, 10)]

label, conf = predict.predict(vis_path, nir_path, model_path, classes)
print(f"Predicted day: {label}, confidence {conf:.2f}")
