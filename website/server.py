from flask import Flask, render_template, request
app = Flask(__name__)

# @app.route('/')
# def hello_world():
#     user_name = request.args.get("userName", "unknown")
#     return render_template('main.html', user=user_name)
# TEST COMMENT

@app.route('/')
def lego():
    return render_template('home.html')

@app.route('/apples')
def applesSite():
    return render_template('apples.html')


guests = ["Dave"]
@app.route('/guest_list', methods=["GET"])
def applesSiteList():
    global guests
    return render_template('guests.html', guests=guests)

@app.route('/guest_list', methods=["POST"])
def guestPost():
    global guests
    guest = request.form.get('guest', 'nobody')
    guests.append(guest)
    return render_template('guests.html', guests=guests)