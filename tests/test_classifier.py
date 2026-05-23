class DummyClassifier:
    def predict(self, image_bytes):
        return {
            "prediction": "cucumber",
            "confidence": 0.99,
            "latency": 0.1
        }