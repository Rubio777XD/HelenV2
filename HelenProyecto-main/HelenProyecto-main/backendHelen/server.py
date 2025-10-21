from flask import Flask, request, jsonify, Response, render_template
from flask_socketio import SocketIO

app = Flask(__name__)
app.config['debug'] = True
# app.config['host'] = 'localhost'
# app.config['port'] = 5000
app.config['SECRET_KEY'] = 'mysecret'
socket = SocketIO()
socket.init_app(app, cors_allowed_origins="*")

@app.get('/')
def index():
	return render_template('index.html')

@app.route('/health', methods=['GET'])
def health():
	data = {'status': 'ok'}
	socket.emit('message', data)
	resp = jsonify(data)
	resp.status_code = 200
	return resp

# @app.route('/user', methods=['GET'])
# def user_data():
# 	data = {
# 		"username": "user",
# 		"email": "test@test.com",
# 	}
# 	resp = jsonify(data)
# 	resp.status_code = 200
# 	return resp

@app.route('/gestures/gesture-key', methods=['POST'])
def publish_gesture_key():
	"""
	Request data format:
	{
		"key": "char",
		"gesture": "gesture_name",
	}
	"""
	data = request.json
	print(data)
	try:
		socket.emit('message', data)
	except:
		return Response(status=500)
	return Response(status=200)




@socket.on('connect')
def connect():
	print('client connected')

@socket.on('message')
def message(data):
	print(f'message: {data}')



if __name__ == '__main__':
	socket.run(app, host='127.0.0.1', port=5000, debug=True)
