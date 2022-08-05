##Tesorboard: Training and Testing plots (errors, accuracy, etc)


from tensorboard import program
import time

logdirNew_2= "logsNew_2/fit/" 

tb3= program.TensorBoard()

tb3.configure(argv=[None, '--logdir', logdirNew_2])

url3 = tb3.launch()


print("-------")
print("ML_New_2")
print(f"Tensorflow listening on {url3}")

time.sleep(100)