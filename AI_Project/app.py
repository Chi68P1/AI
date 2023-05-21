from flask import Flask, render_template, request
from keras.models import load_model
from keras.utils import load_img, img_to_array
import numpy as np
app = Flask(__name__)

dic = {0 : 'Đây là con ong', 1 : 'Đây là con kiến', 2 : 'Đây là con bướm', 3 : 'Đây là con bọ cạp', 4 : 'Đây là con ve sầu', 
5 : 'Đây là con chuồn chuồn', 6 : 'Đây là con bọ hung', 7 : 'Đây là con châu chấu', 8 : 'Đây là con dế mèn', 9 : 'Đây là con nhện', 
10 : 'Đây là con gián',11 : 'Đây là con muỗi', 12 : 'Đây là con bọ rùa', 13 : 'Đây là con bọ ngựa', 14 : 'Đây là con ruồi'}

model = load_model('Predict_insect.h5')

model.make_predict_function()

def predict_label(img_path):
	i = load_img(img_path, target_size=(100,100))
	i = img_to_array(i)/255.0
	i = i.reshape(1, 100,100,3)
	p = model.predict(i)
	predicted_class = np.argmax(p, axis=1)
	return dic[predicted_class[0]]


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Please subscribe  Artificial Intelligence Hub..!!!"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		p = predict_label(img_path)

	return render_template("index.html", prediction = p, img_path = img_path)

if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)
