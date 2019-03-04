import os
from flask import Flask, render_template, request, send_file, redirect, jsonify
import json
cwd = os.getcwd()
import sys
# import datetime
# use folders of generation functions
sys.path.insert(0, cwd + '/CM_generate')
sys.path.insert(0, cwd + '/CM_auxiliary')
sys.path.insert(0, cwd + '/CM_NN_VL')
import CM_GN_harmonise_melody as hrm
import CM_user_output_functions as uof
import zipfile
import shutil

__author__ = 'maxk'

# global harmonisation variables
idiom_name = 'BachChorales'
useGrouping = False
request_code = " "
name_suffix = " "
voiceLeading = "NoVL"

app = Flask(__name__)

APP_ROOT = os.path.dirname(os.path.abspath(__file__))

def zipdir(path, ziph):
    # ziph is zipfile handle
    for root, dirs, files in os.walk(path):
        for file in files:
            ziph.write(os.path.join(root, file))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/", methods=['POST'])
def upload():
    # use globals
    global useGrouping
    global request_code
    global idiom_name
    global name_suffix
    global voiceLeading
    # print('request: ', request)
    if len(request.files.getlist("file")) < 1:
        return
    target = os.path.join(APP_ROOT, 'server_input_melodies/')
    
    # request_code = datetime.datetime.now().strftime("%I_%M_%S%p_%b_%d_%Y")
    # idiom_name = 'BachChorales'
    # useGrouping = True
    if not os.path.isdir(target):
        os.mkdir(target)
    
    # initialise a subfolder name for melodic inputs
    sub_target = 'melodies_'+request_code
    # make a folder to include input melodies
    if not os.path.isdir(target+sub_target):
        os.mkdir(target+sub_target)
    target = target+sub_target+'/'

    # initialise a subfolder for hamonised output IF many requests are given
    output = 'server_harmonised_output/'
    static_output_1 = 'static/harmonisations'
    static_output_2 = 'templates/static/harmonisations'

    for file in request.files.getlist("file"):
        print(file)
        filename = file.filename
        destination = "/".join([target, filename])
        print(destination)
        file.save(destination)

        # parent_dir = os.path.abspath(os.path.join(cwd, os.pardir))
        melodyFolder = target
        # melodyFolder = cwd + '/input_melodies/'
        melodyFileName = filename

        # TODO: makei it properly - this is a temporary vl fix
        tmp_vl_string = 'simple'
        if voiceLeading == 'NoVL':
            print(voiceLeading)
            tmp_vl_string = 'simple'
        elif voiceLeading == 'BBVL':
            print(voiceLeading)
            tmp_vl_string = 'bidirectional_bvl'
        else:
            print('Unknown VL option!')
        print('VL -- ', tmp_vl_string)
        m, idiom = hrm.harmonise_melody_with_idiom(melodyFolder, melodyFileName, idiom_name,targetFolder=output, use_GCT_grouping=useGrouping, voice_leading=tmp_vl_string)
    
    # the output name as produced by chameleon
    initial_output_file_name = m.name+'_'+idiom_name+'.xml'
    # change the name by adding the code
    output_file_name = m.name+name_suffix+'.xml'
    # output_file_name = m.name+'_'+idiom_name+'_'+'grp'+str(int(useGrouping))+'_'+request_code+'.xml'
    os.rename('server_harmonised_output/'+initial_output_file_name, 'server_harmonised_output/'+output_file_name)
    output_file_with_path = 'server_harmonised_output/'+output_file_name

    # also write midi
    midi_name = m.name+name_suffix+'.mid'
    # midi_name = m.name+'_'+idiom_name+'_'+'grp'+str(int(useGrouping))+'_'+request_code+'.mid'
    output_path = os.path.join(APP_ROOT, 'server_harmonised_output/')
    uof.generate_midi(m.output_stream, fileName=midi_name, destination=output_path)
    output_midi_with_path = output_path+'/'+midi_name

    # copy to static folders for playback/display
    shutil.copy2(output_file_with_path, static_output_1)
    shutil.copy2(output_file_with_path, static_output_2)
    # also midi
    shutil.copy2(output_midi_with_path, static_output_1)
    shutil.copy2(output_midi_with_path, static_output_2)

    # # prepare response
    # tmp_json = {}
    # tmp_json['initial_output_file_name'] = initial_output_file_name
    # return jsonify(tmp_json)
    return send_file(filename_or_fp=output_file_with_path, attachment_filename=output_file_name, as_attachment=True)

@app.route("/get_idiom_names", methods=['POST'])
def get_idiom_names():
    idiom_names_list = os.listdir('static/trained_idioms')
    # remove extension
    for i in range( len( idiom_names_list ) ):
        idiom_names_list[i] = idiom_names_list[i].split('.')[0]
    # prepare response
    tmp_json = {}
    tmp_json['idiom_names_list'] = idiom_names_list
    return jsonify(tmp_json)

@app.route("/set_parameters", methods=['POST'])
def set_grouping():
    print('inside set_grouping')
    data = request.get_data()
    dat_json = json.loads(data.decode('utf-8'))
    # use globals
    global useGrouping
    global request_code
    global idiom_name
    global name_suffix
    global voiceLeading
    useGrouping = dat_json['useGrouping']
    request_code = dat_json['clientID']
    idiom_name = dat_json['idiom_name']
    voiceLeading = dat_json['voiceLeading']
    name_suffix = '_'+idiom_name+'_'+'grp'+str(int(useGrouping))+'_'+voiceLeading+'_'+request_code
    print('useGrouping: ', useGrouping)
    print('request_code: ', request_code)
    print('idiom_name: ', idiom_name)
    # prepare response
    tmp_json = {}
    tmp_json['success'] = True
    tmp_json['name_suffix'] = name_suffix
    return jsonify(tmp_json)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8885, debug=True)
