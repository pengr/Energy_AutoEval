from PIL import Image
from absl import app, flags
from concurrent import futures
import os
import json
import shutil

FLAGS = flags.FLAGS

flags.DEFINE_string('dir', "/tmp/", "Dir to convert images")
flags.DEFINE_string('info1', "/tmp/", "label json file")
flags.DEFINE_string('info2', "/tmp/", "to_imagenet json file")


def main(_):
    image_map = {}

    with open(FLAGS.info1, "r") as file:
        json_array = json.load(file)
        for line, i in json_array.items():
            org_path = os.path.join(FLAGS.dir, "/".join(line.split("/")[1:]))
            rename_path = os.path.join(FLAGS.dir, "/".join([line.split("/")[1], "_".join(line.split("/")[1:])]))
            os.rename(org_path, rename_path)
            image_map[rename_path] = str(i[0])

    for path, i in image_map.items():
        if not os.path.exists(os.path.join(FLAGS.dir, i)):
            os.makedirs(os.path.join(FLAGS.dir, i))
        shutil.copy(path, os.path.join(FLAGS.dir, i))
    os.system(f"rm -rf {FLAGS.dir}/ILSVRC*")

    file_map = {}
    with open(FLAGS.info2, "r") as file:

        json_array = json.load(file)
        for i, line in json_array.items():
            file_map[str(i)] = line[0]

    for dir in os.listdir(FLAGS.dir):
        if dir in file_map.keys():
            os.rename(os.path.join(FLAGS.dir, dir), os.path.join(FLAGS.dir, file_map[dir]))


if __name__ == '__main__':
    app.run(main)
