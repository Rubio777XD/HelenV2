labels_dict = {
	0: '0',
    1: '1',
    2: '2', 
    3: '3',
    4: '4',
    5: '5',
    6: '6',
    7: '7',
    8: '8',
    9: '9',
    10: 'Start', #H
    11: 'Clima', #C
    12: 'Foco', #L
    13: 'Ajustes', #A
    14: 'Inicio', #I
	15: 'Dispositivos', #D
    16: 'Reloj' #R
	}

class GestureData:
	def __init__(self, gesture=None, character=None):
		self.gesture = gesture
		self.character = character
