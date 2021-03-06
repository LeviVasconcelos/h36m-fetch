# Human3.6M dataset fetcher

[Human3.6M](http://vision.imar.ro/human3.6m/description.php) is a 3D
human pose dataset containing 3.6 million human poses and corresponding
images. The scripts in this repository make it easy to download,
extract, and preprocess the images and annotations from Human3.6M.

## Requirements

* Python 3
* [`axel`](https://github.com/axel-download-accelerator/axel)
* CDF

Alternatively, a Dockerfile is provided which has all of the
requirements set up. You can use it to run scripts like so:

```bash
$ docker-compose run --rm --user="$(id -u):$(id -g)" main python3 <script>
```

## Usage

1. Firstly, you will need to create an account at
   http://vision.imar.ro/human3.6m/ to gain access to the dataset.
2. Once your account has been approved, log in and inspect your cookies
   to find your PHPSESSID.
3. Copy the configuration file `config.ini.example` to `config.ini`
   and fill in your PHPSESSID.
4. Use the `download_all.py` script to download the dataset,
   `extract_all.py` to extract the downloaded archives, and
   `process_all.py` to preprocess the dataset into an easier to use
   format.

## Frame sampling

Not all frames are selected during the preprocessing step. We assume
that the data will be used in the Protocol #2 setup (see
["Compositional Human Pose Regression"](https://arxiv.org/abs/1704.00159)),
so for subjects S9 and S11 every 64th frame is used. For the training
subjects (S1, S5, S6, S7, and S8), only "interesting" frames are used.
That is, near-duplicate frames during periods of low movement are
skipped.

You can edit `select_frame_indices_to_include()` in `process_all.py` to
change this behaviour.

## License

The code in this repository is licensed under the terms of the
[Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0).

Please read the
[license agreement](http://vision.imar.ro/human3.6m/eula.php) for the
Human3.6M dataset itself, which specifies citations you must make when
using the data in your own research. The file `metadata.xml` is directly
copied from the "Visualisation and large scale prediction software"
bundle from the Human3.6M website, and is subject to the same license
agreement.

## Labeled keypoints:
| id | name | id | name | id | name | id | name |
| -- | ---- | -- | ---- | -- | ---- | -- | ---- |
| 0 | Hips           |  8 | Left Foot     | 16 | Left Shoulder *  | 24 | Right Shoulder *  |
| 1 | Right Up Leg   |  9 | Left Toe Base | 17 | Left Arm        | 25 | Right Arm        |
| 2 | Right Leg      | 10 | Site          | 18 | Left Fore Arm   | 26 | Right Fore Arm   |
| 3 | Right Foot     | 11 | Spine         | 19 | Left Hand -      | 27 | Right Hand  ~     |
| 4 | Right Toe Base | 12 | Spine 1       | 20 | Left Hand Thumb - | 28 | Right Hand Thumb ~ |
| 5 | Site           | 13 | Neck *         | 21 | Site            | 29 | Site             |
| 6 | Left Up Leg    | 14 | Head          | 22 | Left Wrist End + | 30 | Right Wrist End % |
| 7 | Left Leg       | 15 | Site          | 23 | Site           + | 31 | Site            % |
