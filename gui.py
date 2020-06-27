# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'C:/Users/tarun aggarwal/Desktop/sih/sec.ui'
#
# Created by: PyQt5 UI code generator 5.12.3
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
import sys,os


class Ui_MainWindow(object):
    #################################################################################
    def module5(self):
        path='C:/Users/tarun aggarwal/Desktop/New folder/face/'+str(self.label_line.text())+'.jpg'
        os.startfile(path)
    def module4(self):
        path='C:/Users/tarun aggarwal/Desktop/New folder/'+str(self.label_line.text())+'.txt'
        os.startfile(path)
    def module6(self):
        path='C:/Users/tarun aggarwal/Desktop/New folder/unknown/unknown'+str(self.label_line.text())+'.png'
        os.startfile(path)    
    def setupUi3(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(227, 253, 253);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 20, 731, 501))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.label_info = QtWidgets.QLabel(self.frame)
        self.label_info.setGeometry(QtCore.QRect(290, 30, 141, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_info.setFont(font)
        self.label_info.setObjectName("label_info")
        self.label_userid = QtWidgets.QLabel(self.frame)
        self.label_userid.setGeometry(QtCore.QRect(250, 150, 71, 16))
        font = QtGui.QFont()
        font.setPointSize(10)
        self.label_userid.setFont(font)
        self.label_userid.setObjectName("label_userid")
        self.label_line = QtWidgets.QLineEdit(self.frame)
        self.label_line.setGeometry(QtCore.QRect(340, 150, 113, 22))
        self.label_line.setObjectName("label_line")
        self.btn_timelapse = QtWidgets.QPushButton(self.frame)
        self.btn_timelapse.setGeometry(QtCore.QRect(160, 220, 93, 28))
        self.btn_timelapse.setObjectName("btn_timelapse")
        ###########################################################################
        self.btn_timelapse.clicked.connect(self.module4)
        #######################################################################
        self.btn_photo = QtWidgets.QPushButton(self.frame)
        self.btn_photo.setGeometry(QtCore.QRect(290, 220, 93, 28))
        self.btn_photo.setObjectName("btn_photo")
        ###########################################################################
        self.btn_photo.clicked.connect(self.module5)
        self.btn_detail = QtWidgets.QPushButton(self.frame)
        self.btn_detail.setGeometry(QtCore.QRect(430, 220, 93, 28))
        self.btn_detail.setObjectName("btn_detail")
        ##########################################################################
        self.btn_detail.clicked.connect(self.module6)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(0, 0, 120, 51))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.btn_back = QtWidgets.QPushButton(self.frame_3)
        self.btn_back.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.btn_back.setObjectName("btn_back")
        ########################################################################
        self.btn_back.clicked.connect(self.log3)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(620, 10, 111, 41))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.btn_logout = QtWidgets.QPushButton(self.frame_4)
        self.btn_logout.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.btn_logout.setObjectName("btn_logout")
        ########################################################################
        self.btn_logout.clicked.connect(self.mclose)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 26))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi3(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi3(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_info.setText(_translate("MainWindow", "Information"))
        self.label_userid.setText(_translate("MainWindow", "User_id"))
        self.btn_timelapse.setText(_translate("MainWindow", "Timelapse"))
        self.btn_photo.setText(_translate("MainWindow", "Photo"))
        self.btn_detail.setText(_translate("MainWindow", "Unknown"))
        self.btn_back.setText(_translate("MainWindow", "back"))
        self.btn_logout.setText(_translate("MainWindow", "Logout"))
    #################################################3333333333333333333333333333@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
    def module3(self):
        from dataset1 import det
        det(self.User_id.text(),self.lineEdit_2.text())
    def mclose(self):
        self.close()
    def setupUi2(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color:rgb(227,253,253)")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setGeometry(QtCore.QRect(30, 20, 741, 501))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(130, 110, 491, 251))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.label_dataset = QtWidgets.QLabel(self.frame_2)
        self.label_dataset.setGeometry(QtCore.QRect(210, 10, 101, 21))
        font = QtGui.QFont()
        font.setPointSize(14)
        self.label_dataset.setFont(font)
        self.label_dataset.setObjectName("label_dataset")
        self.label_username = QtWidgets.QLabel(self.frame_2)
        self.label_username.setGeometry(QtCore.QRect(140, 50, 81, 16))
        self.label_username.setObjectName("label_username")
        self.User_id = QtWidgets.QLineEdit(self.frame_2)
        self.User_id.setGeometry(QtCore.QRect(230, 50, 111, 22))
        self.User_id.setObjectName("User_id")
        self.label_phoneno = QtWidgets.QLabel(self.frame_2)
        self.label_phoneno.setGeometry(QtCore.QRect(150, 90, 61, 20))
        self.label_phoneno.setObjectName("label_phoneno")
        self.lineEdit_2 = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_2.setGeometry(QtCore.QRect(230, 90, 111, 22))
        self.lineEdit_2.setObjectName("lineEdit_2")
        ##########################################################################
        self.btn_save = QtWidgets.QPushButton(self.frame_2)
        self.btn_save.setGeometry(QtCore.QRect(200, 140, 93, 28))
        self.btn_save.setObjectName("btn_save")
        ############################################################
        self.btn_save.clicked.connect(self.module3)
        self.frame_3 = QtWidgets.QFrame(self.frame)
        self.frame_3.setGeometry(QtCore.QRect(10, 20, 120, 51))
        self.frame_3.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_3.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_3.setObjectName("frame_3")
        self.btn_back = QtWidgets.QPushButton(self.frame_3)
        self.btn_back.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.btn_back.setObjectName("btn_back")
        ###############################################################################
        self.btn_back.clicked.connect(self.log3)
        self.frame_4 = QtWidgets.QFrame(self.frame)
        self.frame_4.setGeometry(QtCore.QRect(630, 10, 101, 41))
        self.frame_4.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_4.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_4.setObjectName("frame_4")
        self.btn_logout = QtWidgets.QPushButton(self.frame_4)
        self.btn_logout.setGeometry(QtCore.QRect(10, 10, 93, 28))
        self.btn_logout.setObjectName("btn_logout")
        #################################################################33
        self.btn_logout.clicked.connect(self.mclose)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi2(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi2(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_dataset.setText(_translate("MainWindow", "Dataset"))
        self.label_username.setText(_translate("MainWindow", "User_Id"))
        self.label_phoneno.setText(_translate("MainWindow", "Name"))
        self.btn_save.setText(_translate("MainWindow", "Save"))
        self.btn_back.setText(_translate("MainWindow", "back"))
        self.btn_logout.setText(_translate("MainWindow", "Logout"))

    ########################################################mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm#####################################
    def module1(self):
        import vedio
    def module2(self):
        import trainning
    def setupUi1(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(800, 600)
        MainWindow.setStyleSheet("background-color: rgb(227, 253, 253);")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.Main_menu = QtWidgets.QFrame(self.centralwidget)
        self.Main_menu.setGeometry(QtCore.QRect(20, -10, 771, 561))
        self.Main_menu.setStyleSheet("background-color: #E3FDFD")
        self.Main_menu.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.Main_menu.setFrameShadow(QtWidgets.QFrame.Raised)
        self.Main_menu.setObjectName("Main_menu")
        self.label_6 = QtWidgets.QLabel(self.Main_menu)
        self.label_6.setGeometry(QtCore.QRect(310, 120, 171, 41))
        font = QtGui.QFont()
        font.setPointSize(17)
        font.setBold(True)
        font.setWeight(75)
        self.label_6.setFont(font)
        self.label_6.setWordWrap(True)
        self.label_6.setObjectName("label_6")
        self.plainTextEdit = QtWidgets.QPlainTextEdit(self.Main_menu)
        self.plainTextEdit.setGeometry(QtCore.QRect(300, 10, 221, 71))
        font = QtGui.QFont()
        font.setFamily("Segoe UI Semibold")
        font.setPointSize(30)
        font.setBold(True)
        font.setWeight(75)
        self.plainTextEdit.setFont(font)
        self.plainTextEdit.setFrameShape(QtWidgets.QFrame.NoFrame)
        self.plainTextEdit.setObjectName("plainTextEdit")
        self.btn_Train = QtWidgets.QPushButton(self.Main_menu)
        self.btn_Train.setGeometry(QtCore.QRect(300, 340, 191, 23))
        self.btn_Train.setObjectName("btn_Train")
        #################################################################################
        self.btn_Train.clicked.connect(self.module2)
        ##################################################################################
        self.btn_FaceRecogniser = QtWidgets.QPushButton(self.Main_menu)
        self.btn_FaceRecogniser.setGeometry(QtCore.QRect(300, 460, 191, 23))
        self.btn_FaceRecogniser.setObjectName("btn_FaceRecogniser")
        ##################################################################################
        self.btn_FaceRecogniser.clicked.connect(self.module1)
        ###################################################################################
        self.dataset = QtWidgets.QPushButton(self.Main_menu)
        self.dataset.setGeometry(QtCore.QRect(310, 240, 171, 23))
        self.dataset.setObjectName("dataset")
        ####################################################################################
        self.dataset.clicked.connect(self.log1)
        ####################################################################################
        
        self.logout = QtWidgets.QPushButton(self.Main_menu)
        self.logout.setGeometry(QtCore.QRect(660, 30, 75, 23))
        self.logout.setObjectName("logout")
        ###################################################################################
        self.logout.clicked.connect(self.mclose)
        ###################################################################################
        self.pushButton_3 = QtWidgets.QPushButton(self.Main_menu)
        self.pushButton_3.setGeometry(QtCore.QRect(14, 30, 121, 23))
        self.pushButton_3.setObjectName("pushButton_3")
        ####################################################################################
        self.pushButton_3.clicked.connect(self.log2)
        ####################################################################################
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 800, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi1(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi1(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_6.setText(_translate("MainWindow", "System Admin"))
        self.plainTextEdit.setPlainText(_translate("MainWindow", "Conqueror"))
        self.btn_Train.setText(_translate("MainWindow", "Train"))
        self.btn_FaceRecogniser.setText(_translate("MainWindow", "FaceRecogniser"))
        self.dataset.setText(_translate("MainWindow", "Dataset"))
        self.logout.setText(_translate("MainWindow", "Logout"))
        self.pushButton_3.setText(_translate("MainWindow", "Information"))
    
    ########################################################mmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmmm#####################################
    def setupUi(self, MainWindow):
        
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(666, 600)
        MainWindow.setStyleSheet("background-color:#30E3CA")
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.centralwidget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.frame = QtWidgets.QFrame(self.centralwidget)
        self.frame.setCursor(QtGui.QCursor(QtCore.Qt.ForbiddenCursor))
        self.frame.setAutoFillBackground(False)
        self.frame.setStyleSheet("background-color:#30E3CA")
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.frame_2 = QtWidgets.QFrame(self.frame)
        self.frame_2.setGeometry(QtCore.QRect(160, 140, 301, 181))
        self.frame_2.setStyleSheet("color: qlineargradient(spread:repeat, x1:0, y1:0, x2:1, y2:1, stop:0 rgba(107, 62, 221, 239), stop:1 rgba(255, 255, 255, 255));")
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.gridLayout = QtWidgets.QGridLayout(self.frame_2)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.label_2 = QtWidgets.QLabel(self.frame_2)
        self.label_2.setStyleSheet("color:#364F6B;")
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_2.addWidget(self.label_2)
        self.lineEdit_password = QtWidgets.QLineEdit(self.frame_2)
        self.lineEdit_password.setStyleSheet("\n"
"Background:-color#E3FDFD;")
        self.lineEdit_password.setInputMethodHints(QtCore.Qt.ImhHiddenText|QtCore.Qt.ImhNoAutoUppercase|QtCore.Qt.ImhNoPredictiveText|QtCore.Qt.ImhSensitiveData)
        self.lineEdit_password.setInputMask("")
        self.lineEdit_password.setText("")
        self.lineEdit_password.setEchoMode(QtWidgets.QLineEdit.Password)
        self.lineEdit_password.setObjectName("lineEdit_password")
        self.horizontalLayout_2.addWidget(self.lineEdit_password)
        self.gridLayout.addLayout(self.horizontalLayout_2, 2, 0, 1, 1)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label = QtWidgets.QLabel(self.frame_2)
        self.label.setStyleSheet("color:#364F6B;")
        self.label.setWordWrap(True)
        self.label.setObjectName("label")
        self.horizontalLayout.addWidget(self.label)
        self.username = QtWidgets.QLineEdit(self.frame_2)
        self.username.setEnabled(True)
        self.username.setStyleSheet("color: rgb(255, 230);\n"
"Background:-color#E3FDFD;")
        self.username.setInputMethodHints(QtCore.Qt.ImhNone)
        self.username.setInputMask("")
        self.username.setText("")
        self.username.setMaxLength(32767)
        self.username.setObjectName("username")
        self.horizontalLayout.addWidget(self.username)
        self.gridLayout.addLayout(self.horizontalLayout, 1, 0, 1, 1)
        self.pushButton_login = QtWidgets.QPushButton(self.frame_2)
        self.pushButton_login.setCursor(QtGui.QCursor(QtCore.Qt.ArrowCursor))
        self.pushButton_login.setStyleSheet("color:#364F6B;\n"
"Background:-color#E3FDFD;")
        self.pushButton_login.setObjectName("pushButton_login")
        self.gridLayout.addWidget(self.pushButton_login, 3, 0, 1, 1)
        ###########################################################################################
        self.pushButton_login.clicked.connect(self.log)
        ###########################################################################################
        self.horizontalLayout_3.addWidget(self.frame)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 666, 21))
        self.menubar.setObjectName("menubar")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)

        self.retranslateUi(MainWindow)
        self.pushButton_login.destroyed.connect(self.frame.close)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)
        MainWindow.setTabOrder(self.username, self.lineEdit_password)
        MainWindow.setTabOrder(self.lineEdit_password, self.pushButton_login)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.label_2.setText(_translate("MainWindow", "Password"))
        self.label.setText(_translate("MainWindow", "User name"))
        self.pushButton_login.setText(_translate("MainWindow", "login"))
        ##################################################################################
    def log(self):
        print(self.username.text(),self.lineEdit_password.text())
        if((self.username.text()=='1') and (self.lineEdit_password.text()=='1')):
            self.close()
            self.window = mainpanelManager1()
            self.window.show()  
    def log3(self):
        self.close()
        self.window = mainpanelManager1()
        self.window.show()
        
    def log1(self):
        self.close()
        self.window = mainpanelManager2()
        self.window.show()        
    def log2(self):
        self.close()
        self.window = mainpanelManager3()
        self.window.show()
               
        ############################################1111111111111111111111111111111111111111################################
class mainpanelManager(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
class mainpanelManager1(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi1(self)
class mainpanelManager2(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi2(self)       
class mainpanelManager3(QtWidgets.QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi3(self)        
if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = mainpanelManager()
    window.show()
    sys.exit(app.exec_())        