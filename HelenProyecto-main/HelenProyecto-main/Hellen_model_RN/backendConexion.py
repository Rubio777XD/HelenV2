import requests
from helpers import labels_dict, GestureData

server_url = 'http://127.0.0.1:5000'
post_server_url = f'{server_url}/gestures/gesture-key'


def post_gesturekey(prediction):
	gesture_name = None
	if __is_gesture_name(prediction):
		gesture_name = prediction
	data = GestureData(gesture_name, prediction).__dict__
	try:
		req = requests.post(post_server_url, json=data)
	except requests.exceptions.RequestException as e:
		print(f'Error Server: {e}')
		return 500
	return req.status_code


def __is_gesture_name(prediction):
    return prediction in labels_dict.values()
