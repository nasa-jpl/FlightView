import sys
from time import sleep
from PyQt4 import QtCore, QtGui, QtNetwork

class saveClient(QtCore.QObject):
	def __init__(self, address, port, parent = None):
		super(saveClient, self).__init__(parent)

		self.blockSize = 0
		self.ipAddress = address
		self.portNumber = port

		#STATUS COMMAND VARIABLES
		self.STATUS_CMD = 3
		self.fps = 0
		self.framesLeft = 0

		#FRAME SAVE COMMAND VARIABLES
		self.FRSAVE_CMD = 2
		self.framesToSave = 0
		self.fname = QtCore.QString("")
		self.numAvgs = 1

		#CREATING A SOCKET WITH CONNECTIONS
		self.socket = QtNetwork.QTcpSocket(self)
		self.socket.error.connect(self.printError)

	def requestFrames(self, file_name, num_frames, n_avgs):
		self.fname = file_name
		self.framesToSave = num_frames
		self.numAvgs = n_avgs
		self.blockSize = 0
		self.socket.abort()

		block = QtCore.QByteArray()
		commStream = QtCore.QDataStream(block, QtCore.QIODevice.WriteOnly)
		commStream.setVersion(QtCore.QDataStream.Qt_4_0)
		commStream.writeUInt16(0)
		commStream.writeUInt16(self.FRSAVE_CMD)
		commStream.writeUInt16(self.framesToSave)
		commStream.writeQString(self.fname)
		commStream.writeUInt16(self.numAvgs)
		commStream.device().seek(0)
		commStream.writeUInt16(block.count() - 2)
		self.socket.connectToHost(self.ipAddress, self.portNumber)
		self.socket.waitForConnected(10)
		self.socket.write(block)
		self.socket.waitForDisconnected()

	def requestStatus(self):
		self.blockSize = 0
		self.socket.abort()

		block = QtCore.QByteArray()
		commStream = QtCore.QDataStream(block, QtCore.QIODevice.WriteOnly)
		commStream.setVersion(QtCore.QDataStream.Qt_4_0)
		commStream.writeUInt16(0)
		commStream.writeUInt16(self.STATUS_CMD)
		commStream.device().seek(0)
		commStream.writeUInt16(block.count() - 2)
		self.socket.connectToHost(self.ipAddress, self.portNumber)
		self.socket.waitForConnected(10)
		self.socket.write(block)
		self.socket.waitForReadyRead()
		inStream = QtCore.QDataStream(self.socket)
		inStream.setVersion(QtCore.QDataStream.Qt_4_0)

		if self.blockSize == 0:
			if self.socket.bytesAvailable() < 2:
				print("No Data Received...")
				return
			self.blockSize = inStream.readUInt16()
		self.framesLeft = inStream.readUInt16()
		self.fps = inStream.readUInt16()
		return (self.framesLeft, self.fps)

	def printError(self,socket_error):
		errors = {
			QtNetwork.QTcpSocket.HostNotFoundError:
				"The host was not found. Please check the host name and "
				"port settings.",
 
			QtNetwork.QTcpSocket.ConnectionRefusedError:
				"The connection was refused by the peer. Make sure that " 
				"Live View is running, and check that the host name and port "
				"settings are correct.",
			
			QtNetwork.QTcpSocket.RemoteHostClosedError:
				None,
		}

		message = errors.get(socket_error,
			"The following error occurred: %s." % self.socket.errorString())
		if message is not None:
			print message

if __name__ == '__main__':
	app = QtGui.QApplication(sys.argv)
	# Replace the ip address and filename as needed.
	client = saveClient("10.0.0.1", 65000)
	#client.requestFrames("/data/test/filename.raw", 10)
	status = client.requestStatus()
	print(status)
