from gradio_client import Client

client = Client("http://127.0.0.1:5050/")
result = client.predict(
	message="Hello!!",
	kb_selector="default",
	api_name="/process_message"
)
print(result)