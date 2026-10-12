from model_builder import build_model, compile_model

model = build_model()
compile_model(model)

model.summary()