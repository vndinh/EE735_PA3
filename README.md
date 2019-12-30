# EE735_PA3
Programming assignment 3, Computer Vision, EE, KAIST, Fall 2018

1. Explanation
	- The './data/cifar-10-batches-py' folder stores the CIFAR10 dataset in binary files
	- The './report/supervised' folder stores the results of supervised learning or baseline
	- The './report/mean_teacher' folder stores the results of mean teacher model
	- The './source' folder consists all coding files
	- The './model/sv_model' folder contains the supervised model
	- The './model/mean_teacher' folder includes the mean teacher model
	- Read the './report/PA3_20184187_DinhVu.pdf' for more details

2. Supervised Learning
	- Copy all files in the './report/supervised/model' folder to the './model/supervised' folder
	- Open Command Promt (Window) or Terminal (Linux) in the './source' folder

	+ If you want to re-train the network, type: python main.py --mode=svtrain
	+ Wait the training process finished
	+ It will take about 25 minutes if your hardware resources is same as mine
	+ It also includes the testing after training

	+ If you only want to test, type: python main.py --mode=svtest
	+ Wait the testing process finished
	+ It will take about 2 minutes if your hardware resources is same as mine

3. Mean Teacher
	- Copy all files in the './report/mean_teacher/model' folder to the './model/mean_teacher' folder
	- Open Command Promt (Window) or Terminal (Linux) in the './source' folder
	
	+ If you want to re-train the network, type: python main.py --mode=mttrain
	+ Wait the training process finished
	+ It also includes the testing after training

	+ If you only want to test, type: python main.py --mode=mttest
	+ Wait the testing process finished



