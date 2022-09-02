import sys
from time import sleep
#from time import sleep
from PyQt5 import QtCore, QtNetwork
#from PyQt4 import QtGui; # only if a GUI is desired

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
        self.numAvgs_stat = 1;
        
        #STATUS_EXTENDED COMMAND VARIABLES
        self.STATUS_CMD_EXTENDED = 4
        self.remoteFilename = str("")

        #FRAME SAVE COMMAND VARIABLES
        self.FRSAVE_CMD = 2
        self.framesToSave = 0
        self.fname = str("")
        #self.fname = QtCore.QString("")
        self.numAvgs = 1

        #CREATING A SOCKET WITH CONNECTIONS
        self.socket = QtNetwork.QTcpSocket(self)
        self.socket.error.connect(self.printError)

    def requestFrames(self, file_name, num_frames, n_avgs=1):
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

    def requestStatus(self, avg_support = True):
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
                print("LVC: No Data Received...")
                return
            self.blockSize = inStream.readUInt16()
        self.framesLeft = inStream.readUInt16()
        self.fps = inStream.readUInt16()
        if avg_support:
            self.numAvgs_stat = inStream.readUInt16(); # new
        else:
            self.numAvgs_stat = 1;

        return (self.framesLeft, self.fps, self.numAvgs_stat);

    def requestStatus_extended(self):
        self.blockSize = 0
        self.socket.abort()

        block = QtCore.QByteArray()
        commStream = QtCore.QDataStream(block, QtCore.QIODevice.WriteOnly)
        commStream.setVersion(QtCore.QDataStream.Qt_4_0)
        commStream.writeUInt16(0)
        commStream.writeUInt16(self.STATUS_CMD_EXTENDED)
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
                print("LVC: No Data Received...")
                return
            self.blockSize = inStream.readUInt16()
        self.framesLeft = inStream.readUInt16()
        self.fps = inStream.readUInt16()
        self.numAvgs_stat = inStream.readUInt16();
        self.remoteFilename = inStream.readQString();
        return (self.framesLeft, self.fps, self.numAvgs_stat, self.remoteFilename);

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
            print(message)

if __name__ == '__main__':
    #app = QtGui.QApplication(sys.argv) # not needed unless you want a GUI
    client = saveClient("127.0.0.1", 65000)
    # Save 1024 frames:
    client.requestFrames("testfile.raw", 1000, 1); # returns immediately
    status_extended = client.requestStatus_extended();
    print("Status extended: " + str(status_extended[0]) + ", " + str(status_extended[1]) + ", " + str(status_extended[2]) + ", " + str(status_extended[3]));
        
    client.requestFrames("/tmp/10_2_socketABCDEFG.raw\00\00", 1000, 1); 
    status_extended = client.requestStatus_extended();
    print("Status extended: " + str(status_extended[0]) + ", " + str(status_extended[1]) + ", " + str(status_extended[2]) + ", " + str(status_extended[3]));
    
    status = client.requestStatus(True)
    estimateRemainingTimeSec = 1.00
    estimateRemainingTimeInitialSec = 1.00
    if(status[2] != 0):
        estimateRemainingTimeInitialSec = status[0] / status[1]
    else:
        estimateRemainingTimeInitialSec = 0
        
    while(status[0] != 0):
        print('(Frames remaining, FPS, averages_per_frame)')
        print(status)
        status = client.requestStatus(True)
        if(status[1] != 0):
            estimateRemainingTimeSec = status[0] / status[1]
        else:
            estimateRemainingTimeSec = 0
        # Update status about ten times per collection
        sleep(estimateRemainingTimeInitialSec / 10)
    # see save_helpers.py for an example on how to wait carefully
