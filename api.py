from flask import Flask, flash, redirect, request,render_template,url_for,session
from werkzeug.utils import secure_filename
import os
from common.util import alignment, face_ToTensor, get_lbpface
from arguments import test_args
from models.models import CreateModel
import torch
import cv2
import numpy as np
from loader import get_loader
from scipy.spatial.distance import cdist
from skimage.feature import local_binary_pattern
from numpy import linalg as la
import base64
from src import detect_faces
from PIL import Image
import csv

_lfw_landmarks = 'data/LFW.csv'
_lfw_images = 'data/peopleDevTest.txt'
_lfw_root = '/home/aaron/Datasets/database/'
_lbpfaces_path = 'data/lbpfaces.npy'
meanface_path = 'data/meanImage.npy'
eigenVec_path = 'data/eigenVectors_new.npy'
weightVec_path = 'data/weightVectors_updated.npy'

args = test_args.get_args()

PEOPLE_FOLDER = os.path.join('static', 'people_photo')
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'jpeg'])

# initialize flask application
app = Flask(__name__, template_folder='templates')
app.config['UPLOAD_FOLDER'] = PEOPLE_FOLDER
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0


frameCount = 0
fileName = ""
filePath = ""


def get_landmarks(image):
    # bb: [x0,y0,x1,y1]
    bounding_boxes, landmarks = detect_faces(image)
    if len(bounding_boxes) > 1:
        # pick the face closed to the center
        center = np.asarray(np.asarray(image).shape[:2]) / 2.0
        ys = bounding_boxes[:, :4][:, [1, 3]].mean(axis=1).reshape(-1, 1)
        xs = bounding_boxes[:, :4][:, [0, 2]].mean(axis=1).reshape(-1, 1)
        coord = np.hstack((ys, xs))
        dist = ((coord - center) ** 2).sum(axis=1)
        index = np.argmin(dist, axis=0)
        landmarks = landmarks[index]
    else:
        landmarks = landmarks[0]
    landmarks = landmarks.reshape(2, 5).T
    landmarks = landmarks.reshape(-1)

    return landmarks


def get_alignedface(image, landmarks):
    img = np.array(image)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    face = alignment(img, landmarks.reshape(-1, 2))
    return face


def get_result(scores):

    with open(_lfw_images) as f:
        images_lines = f.readlines()[1:]

    p = images_lines[int(np.argmax(scores, axis=1))].replace('\n', '').split('\t')
    name_1 = p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    face_1 = cv2.imread(_lfw_root + name_1)
    score_1 = "{0:.2f}".format(np.max(scores)*100)

    scores = np.delete(scores, np.argmax(scores, axis=1))
    p = images_lines[int(np.argmax(scores))].replace('\n', '').split('\t')
    name_2 = p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    face_2 = cv2.imread(_lfw_root + name_2)
    score_2 = "{0:.2f}".format(np.max(scores)*100)

    scores = np.delete(scores, np.argmax(scores))
    p = images_lines[int(np.argmax(scores))].replace('\n', '').split('\t')
    name_3 = p[0] + '_' + '{:04}.jpg'.format(int(p[1]))
    face_3 = cv2.imread(_lfw_root + name_3)
    score_3 = "{0:.2f}".format(np.max(scores)*100)

    return face_1, face_2, face_3, name_1, name_2, name_3, score_1, score_2, score_3


def evaluation(net, face):
    dataloader = get_loader(batch_size=128).dataloader
    features_total = torch.Tensor(np.zeros((args.num_faces, 512), dtype=np.float32)).to(args.device)
    labels = torch.Tensor(np.zeros((args.num_faces, 1), dtype=np.float32)).to(args.device)
    with torch.no_grad():
        bs_total = 0
        for index, (img, targets) in enumerate(dataloader):
            bs = len(targets)
            img = img.to(args.device)
            features = net(img)
            features_total[bs_total:bs_total + bs] = features
            labels[bs_total:bs_total + bs] = targets
            bs_total += bs
        assert bs_total == args.num_faces, print('Database should have {} faces!'.format(args.num_faces))

    input_feature = net(face_ToTensor(face).to(args.device).view([1, 3, 112, 96]))

    input_feature = input_feature.cpu().detach().numpy()
    features_total = features_total.cpu().detach().numpy()
    scores = 1 - cdist(input_feature, features_total, 'cosine')

    return get_result(scores)


def lbp_evaluation(face):
    lbpface = get_lbpface(face)
    lbpface = lbpface.reshape([1, 256])
    lbpfaces = np.load(_lbpfaces_path)
    scores = 1 - cdist(lbpface, lbpfaces, 'cosine')
    return get_result(scores)


def eigenface_evaluation(face):
    meanImage = np.load(meanface_path)
    eigenVecs = np.load(eigenVec_path)
    weightVecs = np.load(weightVec_path)
    '''
    Normalize the eigenvector so its norm is 1
    '''
    for i in range(len(eigenVecs)):
        eigenVecs[i] /= la.norm(eigenVecs[i])
    face = (face.flatten() - meanImage).astype('int8')
    weiVec_face = np.dot(face, eigenVecs.T).reshape([1, -1])
    scores = 1 - cdist(weiVec_face, weightVecs, 'cosine')
    reconFace = meanImage
    for i in range(100):
        '''
        The weight is the dot product of the mean subtracted
        image vector with the EigenVector
        '''
        weight = np.dot(face, eigenVecs[i])
        reconFace = reconFace + eigenVecs[i] * weight
    reconFace = reconFace.reshape([112, 96, 3])
    reconFace = cv2.normalize(reconFace, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)

    return reconFace, get_result(scores)


def allowed_file(filename):
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/")
def gui():
    return render_template('gui.html')


@app.route("/register", methods=['GET', 'POST'])
def register():
    if request.method == 'POST':

        name = request.form['name']
        name = name.replace(' ', '_')
        if name in open(_lfw_images).read():
            return "Sorry, the face has already been registered!"
        else:
            data_url = request.form['data']
            content = data_url.split(';')[1]
            image_encoded = content.split(',')[1]
            body = base64.decodebytes(image_encoded.encode('utf-8'))
            file_name = name + '_0001.jpg'
            save_path = os.path.join(_lfw_root, file_name)
            with open(save_path, "wb") as fh:
                fh.write(body)
            image = Image.open(save_path)
            landmarks = get_landmarks(image)
            # landmark_list = landmarks.tolist().insert(0, name + '/' + file_name)
            landmark_list = landmarks.tolist()
            landmark_list.insert(0, name + '/' + file_name)
            with open(_lfw_landmarks, 'a') as fd:
                writer = csv.writer(fd)
                writer.writerow(landmark_list)
            with open(_lfw_images, "a") as f:
                f.write('\n' + name + '\t' + '1')

            aligned_new_face = get_alignedface(image, landmarks)
            """
            Append lbp histogram of the new face to the array
            """
            lbpface = get_lbpface(aligned_new_face)
            lbpface = lbpface.reshape([1, 256])
            lbpfaces = np.load(_lbpfaces_path)
            lbpfaces = np.vstack((lbpfaces, lbpface))
            np.save(_lbpfaces_path, lbpfaces)

            '''
            Append pca weight of the new face to the array
            '''
            meanImage = np.load(meanface_path)
            eigenVecs = np.load(eigenVec_path)
            weightVecs = np.load(weightVec_path)
            for i in range(len(eigenVecs)):
                eigenVecs[i] /= la.norm(eigenVecs[i])
            face = (aligned_new_face.flatten() - meanImage).astype('int8')
            weiVec_new_face = np.dot(face, eigenVecs.T).reshape([1, -1])
            weightVecs = np.vstack((weightVecs, weiVec_new_face))
            np.save(weightVec_path, weightVecs)

            args.num_faces += 1
            return "Registration Done"
    else:
        return redirect(request.url)


@app.route("/result_cam")
def result_cam():
    use_eigenface = session['use_eigenface']
    use_lbp = session['use_lbp']
    webcam_alignedimg_path = session['aligned_face']
    face_1_path = session['image1']
    name1 = session['name1']
    score1 = session['score1']
    face_2_path = session['image2']
    name2 = session['name2']
    score2 = session['score2']
    face_3_path = session['image3']
    name3 = session['name3']
    score3 = session['score3']
    pcaselect = session['pcaselect']
    lbpselect = session['lbpselect']
    model = session['model']
    loss = session['loss']

    if use_eigenface:
        mean_face_path = session['mean_face_path']
        eigenface1 = session['eigenface1']
        eigenface2 = session['eigenface2']
        eigenface3 = session['eigenface3']
        eigenface4 = session['eigenface4']
        eigenface5 = session['eigenface5']
        eigenface6 = session['eigenface6']
        eigenface7 = session['eigenface7']
        eigenface8 = session['eigenface8']
        eigenface9 = session['eigenface9']
        recon_face_path = session['recon_face_path']

        return render_template('pca_result_cam.html',
                               aligned_face=webcam_alignedimg_path,
                               image1=face_1_path, name1=name1, score1=score1,
                               image2=face_2_path, name2=name2, score2=score2,
                               image3=face_3_path, name3=name3, score3=score3,
                               mean_face=mean_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                               model=model, loss=loss, eigenface1=eigenface1, eigenface2=eigenface2,
                               eigenface3=eigenface3, eigenface4=eigenface4, eigenface5=eigenface5,
                               eigenface6=eigenface6, eigenface7=eigenface7, eigenface8=eigenface8,
                               eigenface9=eigenface9, recon_face=recon_face_path, num_faces=args.num_faces)
    elif use_lbp:
        lbp_face_path = session['lbp_face']
        return render_template('lbp_result_cam.html', aligned_face=webcam_alignedimg_path,
                               image1=face_1_path, name1=name1, score1=score1,
                               image2=face_2_path, name2=name2, score2=score2,
                               image3=face_3_path, name3=name3, score3=score3,
                               lbp_face=lbp_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                               model=model, loss=loss, num_faces=args.num_faces)
    else:
        return render_template('net_result_cam.html', aligned_face=webcam_alignedimg_path,
                               image1=face_1_path, name1=name1, score1=score1,
                               image2=face_2_path, name2=name2, score2=score2,
                               image3=face_3_path, name3=name3, score3=score3,
                               pcaselect=pcaselect, lbpselect=lbpselect, model=model, loss=loss, num_faces=args.num_faces)


@app.route('/camresult', methods=['GET', 'POST'])
def camresult():
    global frameCount
    if request.method == 'POST':
        if request.form.get('pcaselect'):
            use_eigenface = True
            use_lbp = False
            model = "none"
            loss = "none"
            pcaselect = "yes"
            lbpselect = "no"
        elif request.form.get('lbpselect'):
            use_eigenface = False
            use_lbp = True
            model = "none"
            loss = "none"
            pcaselect = "no"
            lbpselect = "yes"
        else:
            use_eigenface = False
            use_lbp = False
            pcaselect = "no"
            lbpselect = "no"
            model = request.form['model']
            loss = request.form['loss']
            model_pth = 'backbone/{}_{}.pth'.format(model, loss)
            if model == '10':
                args.backbone = 'spherenet10'
            elif model == '20':
                args.backbone = 'spherenet20'
            elif model == '64':
                args.backbone = 'spherenet64'
            netModel = CreateModel(args)
            netModel.backbone.load_state_dict(torch.load(model_pth))

        data_url = request.form['data']
        content = data_url.split(';')[1]
        image_encoded = content.split(',')[1]
        body = base64.decodebytes(image_encoded.encode('utf-8'))
        webcam_img_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_img.jpg')
        with open(webcam_img_path, "wb") as fh:
            fh.write(body)
        image = Image.open(webcam_img_path)
        landmarks = get_landmarks(image)
        aligned_face = get_alignedface(image, landmarks)

        webcam_alignedimg_path = os.path.join(app.config['UPLOAD_FOLDER'], 'webcam_alignedimg.jpg')
        cv2.imwrite(webcam_alignedimg_path, aligned_face)
        if use_eigenface:
            recon_face, eigenface_result = eigenface_evaluation(aligned_face)
            recon_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstructed_face.jpg')
            cv2.imwrite(recon_face_path, recon_face)
            face_1, face_2, face_3,  name_1, name_2, name_3, score_1, score_2, score_3 = eigenface_result
        elif use_lbp:
            lbp_face = local_binary_pattern(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY), 8, 1)
            lbp_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lbp_face.jpg')
            cv2.imwrite(lbp_face_path, lbp_face)
            face_1, face_2, face_3,  name_1, name_2, name_3, score_1, score_2, score_3 = lbp_evaluation(aligned_face)
        else:
            face_1, face_2, face_3,  name_1, name_2, name_3, score_1, score_2, score_3 = evaluation(netModel.backbone, aligned_face)

        face_1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_1.jpg')
        face_2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_2.jpg')
        face_3_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_3.jpg')
        cv2.imwrite(face_1_path, face_1)
        cv2.imwrite(face_2_path, face_2)
        cv2.imwrite(face_3_path, face_3)
        name1 = name_1.rsplit('_', 1)[0]
        name2 = name_2.rsplit('_', 1)[0]
        name3 = name_3.rsplit('_', 1)[0]

        session['use_eigenface'] = use_eigenface
        session['use_lbp'] = use_lbp
        session['aligned_face'] = webcam_alignedimg_path
        session['image1'] = face_1_path
        session['name1'] = name1
        session['score1'] = score_1
        session['image2'] = face_2_path
        session['name2'] = name2
        session['score2'] = score_2
        session['image3'] = face_3_path
        session['name3'] = name3
        session['score3'] = score_3
        session['pcaselect'] = pcaselect
        session['lbpselect'] = lbpselect
        session['model'] = model
        session['loss'] = loss
        if use_eigenface:
            session['mean_face_path'] = os.path.join(app.config['UPLOAD_FOLDER'], 'meanface.jpg')
            session['eigenface1'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface1.jpg')
            session['eigenface2'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface2.jpg')
            session['eigenface3'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface3.jpg')
            session['eigenface4'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface4.jpg')
            session['eigenface5'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface5.jpg')
            session['eigenface6'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface6.jpg')
            session['eigenface7'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface7.jpg')
            session['eigenface8'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface8.jpg')
            session['eigenface9'] = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface9.jpg')
            session['recon_face_path'] = recon_face_path
            return redirect(url_for('result_cam'))
        elif use_lbp:
            session['lbp_face'] = lbp_face_path
            return redirect(url_for('result_cam'))
        else:
            return redirect(url_for('result_cam'))
    else:
        return redirect(request.url)
        # return "OK"


@app.route('/result', methods=['GET', 'POST'])
def handle_data():
    global frameCount
    global filePath
    global fileName
    if request.method == 'POST':
        if request.form.get('pcaselect'):
            use_eigenface = True
            use_lbp = False
            model = "none"
            loss = "none"
            pcaselect = "yes"
            lbpselect = "no"
        elif request.form.get('lbpselect'):
            use_eigenface = False
            use_lbp = True
            model = "none"
            loss = "none"
            pcaselect = "no"
            lbpselect = "yes"
        else:
            use_eigenface = False
            use_lbp = False
            pcaselect = "no"
            lbpselect = "no"
            model = request.form['model']
            loss = request.form['loss']
            model_pth = 'backbone/{}_{}.pth'.format(model, loss)

            if model == '10':
                args.backbone = 'spherenet10'
            elif model == '20':
                args.backbone = 'spherenet20'
            elif model == '64':
                args.backbone = 'spherenet64'
            netModel = CreateModel(args)
            netModel.backbone.load_state_dict(torch.load(model_pth))
        if 'file' not in request.files:
            if frameCount > 1:
                image = Image.open(filePath)
                landmarks = get_landmarks(image)
                aligned_face = get_alignedface(image, landmarks)

                aligned_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face.jpg')
                cv2.imwrite(aligned_face_path, aligned_face)
                if use_eigenface:
                    recon_face, eigenface_result = eigenface_evaluation(aligned_face)
                    recon_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstructed_face.jpg')
                    cv2.imwrite(recon_face_path, recon_face)
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = eigenface_result
                elif use_lbp:
                    lbp_face = local_binary_pattern(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY), 8, 1)
                    lbp_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lbp_face.jpg')
                    cv2.imwrite(lbp_face_path, lbp_face)
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = lbp_evaluation(aligned_face)
                else:
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = evaluation(netModel.backbone, aligned_face)
                face_1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_1.jpg')
                face_2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_2.jpg')
                face_3_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_3.jpg')
                cv2.imwrite(face_1_path, face_1)
                cv2.imwrite(face_2_path, face_2)
                cv2.imwrite(face_3_path, face_3)
                imageName = fileName.rsplit('_', 1)[0]
                name1 = name1.rsplit('_', 1)[0]
                name2 = name2.rsplit('_', 1)[0]
                name3 = name3.rsplit('_', 1)[0]

                if use_eigenface:
                    mean_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meanface.jpg')
                    eigenface1 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface1.jpg')
                    eigenface2 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface2.jpg')
                    eigenface3 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface3.jpg')
                    eigenface4 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface4.jpg')
                    eigenface5 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface5.jpg')
                    eigenface6 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface6.jpg')
                    eigenface7 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface7.jpg')
                    eigenface8 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface8.jpg')
                    eigenface9 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface9.jpg')

                    return render_template('pca_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           mean_face=mean_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                           model=model, loss=loss, eigenface1=eigenface1, eigenface2=eigenface2,
                                           eigenface3=eigenface3, eigenface4=eigenface4, eigenface5=eigenface5,
                                           eigenface6=eigenface6, eigenface7=eigenface7, eigenface8=eigenface8,
                                           eigenface9=eigenface9, recon_face=recon_face_path, num_faces=args.num_faces)
                elif use_lbp:
                    return render_template('lbp_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           lbp_face=lbp_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                           model=model, loss=loss, num_faces=args.num_faces)
                else:
                    return render_template('net_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           pcaselect=pcaselect, lbpselect=lbpselect, model=model, loss=loss, num_faces=args.num_faces)

            else:
                flash('No file part')
                return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            if frameCount > 1:

                image = Image.open(filePath)
                landmarks = get_landmarks(image)
                aligned_face = get_alignedface(image, landmarks)
                aligned_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face.jpg')

                cv2.imwrite(aligned_face_path, aligned_face)
                if use_eigenface:
                    recon_face, eigenface_result = eigenface_evaluation(aligned_face)
                    recon_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstructed_face.jpg')
                    cv2.imwrite(recon_face_path, recon_face)
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = eigenface_result
                elif use_lbp:
                    lbp_face = local_binary_pattern(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY), 8, 1)
                    lbp_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lbp_face.jpg')
                    cv2.imwrite(lbp_face_path, lbp_face)
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = lbp_evaluation(aligned_face)
                else:
                    face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = evaluation(netModel.backbone, aligned_face)
                face_1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_1.jpg')
                face_2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_2.jpg')
                face_3_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_3.jpg')
                cv2.imwrite(face_1_path, face_1)
                cv2.imwrite(face_2_path, face_2)
                cv2.imwrite(face_3_path, face_3)
                imageName = fileName.rsplit('_', 1)[0]
                name1 = name1.rsplit('_', 1)[0]
                name2 = name2.rsplit('_', 1)[0]
                name3 = name3.rsplit('_', 1)[0]
                # return '', 204
                # return render_template('gui.html', accuracy='2', image=file_path)
                if use_eigenface:
                    mean_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meanface.jpg')
                    eigenface1 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface1.jpg')
                    eigenface2 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface2.jpg')
                    eigenface3 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface3.jpg')
                    eigenface4 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface4.jpg')
                    eigenface5 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface5.jpg')
                    eigenface6 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface6.jpg')
                    eigenface7 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface7.jpg')
                    eigenface8 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface8.jpg')
                    eigenface9 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface9.jpg')

                    return render_template('pca_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           mean_face=mean_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                           model=model, loss=loss, eigenface1=eigenface1, eigenface2=eigenface2,
                                           eigenface3=eigenface3, eigenface4=eigenface4, eigenface5=eigenface5,
                                           eigenface6=eigenface6, eigenface7=eigenface7, eigenface8=eigenface8,
                                           eigenface9=eigenface9, recon_face=recon_face_path, num_faces=args.num_faces)
                elif use_lbp:
                    return render_template('lbp_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           lbp_face=lbp_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                           model=model, loss=loss, num_faces=args.num_faces)
                else:
                    return render_template('net_result.html', imageName=imageName, input_face=filePath,
                                           aligned_face=aligned_face_path,
                                           image1=face_1_path, name1=name1, score1=score1,
                                           image2=face_2_path, name2=name2, score2=score2,
                                           image3=face_3_path, name3=name3, score3=score3,
                                           pcaselect=pcaselect, lbpselect=lbpselect, model=model, loss=loss, num_faces=args.num_faces)
                # return '', 204
            else:
                flash('No selected file')
                return redirect(request.url)
        if file and allowed_file(file.filename):
            frameCount = frameCount + 1
            file_name = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], file_name)
            file.save(file_path)
            filePath = file_path
            fileName = file_name
            image = Image.open(file_path)
            landmarks = get_landmarks(image)
            aligned_face = get_alignedface(image, landmarks)

            aligned_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'aligned_face.jpg')
            cv2.imwrite(aligned_face_path, aligned_face)
            if use_eigenface:
                recon_face, eigenface_result = eigenface_evaluation(aligned_face)
                recon_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'reconstructed_face.jpg')
                cv2.imwrite(recon_face_path, recon_face)
                face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = eigenface_result
            elif use_lbp:
                lbp_face = local_binary_pattern(cv2.cvtColor(aligned_face, cv2.COLOR_BGR2GRAY), 8, 1)
                lbp_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'lbp_face.jpg')
                cv2.imwrite(lbp_face_path, lbp_face)
                face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = lbp_evaluation(aligned_face)
            else:
                face_1, face_2, face_3, name1, name2, name3, score1, score2, score3 = evaluation(netModel.backbone, aligned_face)
            face_1_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_1.jpg')
            face_2_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_2.jpg')
            face_3_path = os.path.join(app.config['UPLOAD_FOLDER'], 'face_3.jpg')
            cv2.imwrite(face_1_path, face_1)
            cv2.imwrite(face_2_path, face_2)
            cv2.imwrite(face_3_path, face_3)
            imageName = file_name.rsplit('_', 1)[0]
            name1 = name1.rsplit('_', 1)[0]
            name2 = name2.rsplit('_', 1)[0]
            name3 = name3.rsplit('_', 1)[0]
            # return '', 204
            # return render_template('gui.html', accuracy='2', image=file_path)
            if use_eigenface:
                mean_face_path = os.path.join(app.config['UPLOAD_FOLDER'], 'meanface.jpg')
                eigenface1 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface1.jpg')
                eigenface2 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface2.jpg')
                eigenface3 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface3.jpg')
                eigenface4 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface4.jpg')
                eigenface5 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface5.jpg')
                eigenface6 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface6.jpg')
                eigenface7 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface7.jpg')
                eigenface8 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface8.jpg')
                eigenface9 = os.path.join(app.config['UPLOAD_FOLDER'], 'eigenface9.jpg')

                return render_template('pca_result.html', imageName=imageName, input_face=filePath,
                                       aligned_face=aligned_face_path,
                                       image1=face_1_path, name1=name1, score1=score1,
                                       image2=face_2_path, name2=name2, score2=score2,
                                       image3=face_3_path, name3=name3, score3=score3,
                                       mean_face=mean_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                       model=model, loss=loss, eigenface1=eigenface1, eigenface2=eigenface2,
                                       eigenface3=eigenface3, eigenface4=eigenface4, eigenface5=eigenface5,
                                       eigenface6=eigenface6, eigenface7=eigenface7, eigenface8=eigenface8,
                                       eigenface9=eigenface9, recon_face=recon_face_path, num_faces=args.num_faces)
            elif use_lbp:
                return render_template('lbp_result.html', imageName=imageName, input_face=file_path,
                                       aligned_face=aligned_face_path,
                                       image1=face_1_path, name1=name1, score1=score1,
                                       image2=face_2_path, name2=name2, score2=score2,
                                       image3=face_3_path, name3=name3, score3=score3,
                                       lbp_face=lbp_face_path, pcaselect=pcaselect, lbpselect=lbpselect,
                                       model=model, loss=loss, num_faces=args.num_faces)
            else:
                return render_template('net_result.html', imageName=imageName, input_face=file_path,
                                       aligned_face=aligned_face_path,
                                       image1=face_1_path, name1=name1, score1=score1,
                                       image2=face_2_path, name2=name2, score2=score2,
                                       image3=face_3_path, name3=name3, score3=score3,
                                       pcaselect=pcaselect, lbpselect=lbpselect, model=model, loss=loss, num_faces=args.num_faces)
    else:
        return redirect(request.url)


# No caching at all for API endpoints.
@app.after_request
def add_header(response):
    # response.cache_control.no_store = True
    response.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, post-check=0, pre-check=0, max-age=0'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '-1'
    return response


app.secret_key = "super secret key"
if __name__ == '__main__':
    with open(_lfw_images) as f:
        size = sum(1 for _ in f) - 1
    args.num_faces = size
    app.run(host='0.0.0.0', port=8082)
