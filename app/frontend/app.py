import gradio as gr
import requests

BACKEND_URL = "http://127.0.0.1:8000/predict"

def classify_image(image):
    if image is None:
        return "Please upload an image", None

    # Convert image to bytes
    import io
    img_bytes = io.BytesIO()
    image.save(img_bytes, format="JPEG")
    img_bytes = img_bytes.getvalue()

    try:
        response = requests.post(
            BACKEND_URL,
            files={"file": ("image.jpg", img_bytes, "image/jpeg")}
        )

        if response.status_code == 200:
            data = response.json()
            prediction = data.get("prediction", "Unknown")
            confidence = data.get("confidence", "N/A")
            latency = data.get("latency", "N/A")

            return f"Prediction: {prediction}", f"Confidence: {confidence:.4f}", f"Latency: {latency:.4f}"

        else:
            return f"Error: {response.json().get('detail')}", "", ""

    except Exception as e:
        return f"Connection error: {str(e)}", "", ""


iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="pil"),
    outputs=[
        gr.Textbox(label="Prediction"),
        gr.Textbox(label="Confidence"),
        gr.Textbox(label="Latency (s)")
    ],
    title="Vegetable Classifier",
    description="Upload an image to classify vegetables using your ML backend.",
)

if __name__ == "__main__":
    # iface.launch() # default address is 127.0.0.1:7860
    # iface.launch(share=True)
    iface.launch(server_name="0.0.0.0", server_port=7860)