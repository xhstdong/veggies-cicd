class DummyClassifier:
    def predict(self, image_bytes):
        return {
            "class": "Cucumber",
            "confidence": 0.99
        }