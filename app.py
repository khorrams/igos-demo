from flask import Flask, render_template, request
from flask import jsonify
import dummy
from Generate_video_Final import gen_video
from grad_cam import gcam
app = Flask(__name__)


@app.route("/")
def home():
    return render_template("index.html")


@app.route('/metricIGOS', methods=['POST'])
def IGOS():
    # get the url
    url = request.form['url']

    # if from carousel
    if url.startswith('$%#carousel'):

        # get the local image path
        image_path = dummy.carousel_name(url)
        print(image_path)

    # if user enters a url
    else:
        try:
            # download the image
            image_path = dummy.video_image_name(url)
            print(image_path)

        # in case the image could not be downloaded
        except:

            return 'Something went Wrong! Please check the URL and try again :)'

    # Try generate a saleincy map and send it back to the front-end
    try:

        # GradCam saliency map
        gcam_mask, gcam_dir = gcam(image_path)
        print('gcam info', gcam_mask.shape, gcam_dir)

        # get the insertion and deletion score for GradCam and I-GOS
        ins_c, del_c, ins_c_gcam, del_c_gcam, out_dir, cat_name = gen_video(input_img_path=image_path,
                                                                            mask_gcam=gcam_mask,
                                                                            output_path_gcam=gcam_dir)

        # create a msg with desired information and return it
        msg = jsonify({'ins_c': ins_c, 'del_c': del_c, 'ins_c_gcam': ins_c_gcam, 'del_c_gcam': del_c_gcam,
                       'out_dir': out_dir, 'cat_name': cat_name, 'gcam_dir': gcam_dir})
        return msg

    # in case the mask generation ran into error
    except:
        return 'Unexpected error generating the video! Please try again with another image :)'
    # return jsonify({'video_url': '/static/videos/69.mp4'})


@app.errorhandler(404)
def page_not_found(error):
    app.logger.error('Page not found: %s', (request.path))
    return render_template('404.htm'), 404


@app.errorhandler(500)
def internal_server_error(error):
    app.logger.error('Server Error: %s', (error))
    return render_template('500.htm'), 500


if __name__ == "__main__":
    app.run()
