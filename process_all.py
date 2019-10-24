#!/usr/bin/env python3
import sys
import os
from os import path, makedirs, listdir
from shutil import move
from spacepy import pycdf
import numpy as np
import h5py
from subprocess import call
from tempfile import TemporaryDirectory
from tqdm import tqdm
import cv2
from metadata import load_h36m_metadata
from PIL import Image

metadata = load_h36m_metadata()

# Subjects to include when preprocessing
included_subjects = {
    'S1': 1,
    'S5': 5,
    'S6': 6,
    'S7': 7,
    'S8': 8,
    'S9': 9,
    'S11': 11,
}

# Sequences with known issues
blacklist = {
    ('S11', '2', '2', '54138969'),  # Video file is corrupted
    ('S7', '15', '2', ''), # TOF video does not exists.
    ('S5', '4', '2', ''), # TOF video does not exists.
}


# Rather than include every frame from every video, we can instead wait for the pose to change
# significantly before storing a new example.
def select_frame_indices_to_include(subject, poses_3d_univ):
    # To process every single frame, uncomment the following line:
    # return np.arange(0, len(poses_3d_univ))

    # Take every 64th frame for the protocol #2 test subjects
    # (see the "Compositional Human Pose Regression" paper)
    if subject == 'S9' or subject == 'S11':
        return np.arange(0, len(poses_3d_univ), 64)

    # Take only frames where movement has occurred for the protocol #2 train subjects
    frame_indices = []
    prev_joints3d = None
    threshold = 40 ** 2  # Skip frames until at least one joint has moved by 40mm
    for i, joints3d in enumerate(poses_3d_univ):
        if prev_joints3d is not None:
            max_move = ((joints3d - prev_joints3d) ** 2).sum(axis=-1).max()
            if max_move < threshold:
                continue
        prev_joints3d = joints3d
        frame_indices.append(i)
    return np.array(frame_indices)


def infer_camera_intrinsics(points2d, points3d):
    """Infer camera instrinsics from 2D<->3D point correspondences."""
    pose2d = points2d.reshape(-1, 2)
    pose3d = points3d.reshape(-1, 3)
    x3d = np.stack([pose3d[:, 0], pose3d[:, 2]], axis=-1)
    x2d = (pose2d[:, 0] * pose3d[:, 2])
    alpha_x, x_0 = list(np.linalg.lstsq(x3d, x2d, rcond=-1)[0].flatten())
    y3d = np.stack([pose3d[:, 1], pose3d[:, 2]], axis=-1)
    y2d = (pose2d[:, 1] * pose3d[:, 2])
    alpha_y, y_0 = list(np.linalg.lstsq(y3d, y2d, rcond=-1)[0].flatten())
    return np.array([alpha_x, x_0, alpha_y, y_0])

def _reshape_tof(tof_stack):
      'Reshape into n_views x 144 x 176'
      tmp = np.dsplit(tof_stack, tof_stack.shape[2])
      tmp = [p.reshape(p.shape[0], p.shape[1]) for p in tmp]
      return np.stack(tmp)

def _decode_tof_intensity(tof_int, threshold=2300):
      intensity = np.where(tof_int > threshold, threshold, tof_int)
      intensity = cv2.convertScaleAbs(intensity, alpha=255/np.max(intensity))
      return intensity

def _process_bbox(bbox_mask):
      rows = np.any(bbox_mask, axis=1)
      cols = np.any(bbox_mask, axis=0)
      rmin, rmax = np.where(rows)[0][[0, -1]]
      cmin, cmax = np.where(cols)[0][[0, -1]]
      return np.array([[cmin, rmin], [cmax, rmax]])

def _get_3D_BBox(pose3D):
      min_coords = np.min(pose3D, axis=0)
      max_coords = np.max(pose3D, axis=0)
      return [min_coords, max_coords]

def _make_zero_center(pose3D):
      center = np.mean(pose3D, axis=0)
      return pose3D - center

def _make_unit_diagonal(pose3D):
      bbox_min, bbox_max = _get_3D_BBox(pose3D)
      length = np.linalg.norm((bbox_max - bbox_min))
      return pose3D / (length)
      
def _make_centred_unit_bbox(pose3D):
      centred_pose = _make_centred_unit_bbox(pose3D)
      return _make_unit_diagonal(centred_pose)

def _compute_scales(old_size, new_size):
      nrows, ncols = new_size
      fy = nrows / old_size[0]
      fx = ncols / old_size[1]
      return fx,fy
      

def _draw_annot(img, bbox, pose):
      img2 = cv2.copyTo(img, None)
      #cv2.rectangle(img2, tuple(bbox[0]), tuple(bbox[1]), (0, 255, 0), 3)
      for i in pose:
            cv2.circle(img2, tuple(i), 1, (255,0,0), -1)
      return img2

def _pad_with_zeros(img, new_res):
    y_gap = new_res[0] - img.shape[0]
    x_gap = new_res[1] - img.shape[1]
    if (x_gap < 0 or y_gap < 0):
          return img
    new_img = np.zeros(new_res, dtype=img.dtype)
    x_gap = x_gap // 2
    y_gap = y_gap // 2
    new_img[y_gap:(img.shape[0] + y_gap), x_gap:(img.shape[1] + x_gap)] = img.copy()
    return new_img
      

def _make_square(bbox):
    origin_box = bbox[1] - bbox[0]
    min_idx = np.argmin(origin_box)
    max_idx = np.argmax(origin_box)
    diff = (origin_box[max_idx] - origin_box[min_idx])
    resize = np.zeros((2,2), dtype=np.float)
    resize[:, min_idx] = np.asarray([-diff/2, diff/2])
    new_bbox = bbox + resize
    return new_bbox

def _transform_image(image, bbox_):
    img_size = 128
    bbox = _make_square(bbox_)
    ratio = float(image.size[1] / img_size)
    box = tuple(bbox.reshape(-1))
    image_ = image.crop(box=box).resize((img_size, img_size))
    if image_ is None:
        print('Image none, cropbox: ', box)
        print('Image shape: ', image.size)
        raise ValueError 
   
    return image_ 

def _transform_2dkp_annots(kps_, bbox_):
    img_size = 128
    bbox = _make_square(bbox_)
    bbox_size = bbox[1][0] - bbox[0][0]
    ratio = float(bbox_size / img_size)
    shifted_kps = kps_ - bbox[0]
    scaled_kps = shifted_kps / ratio

    return scaled_kps




def process_view(out_dir, subject, action, subaction, camera):
    subj_dir = path.join('extracted', subject)

    base_filename = metadata.get_base_filename(subject, action, subaction, camera)
    tof_filename = metadata.get_base_filename(subject, action, subaction, '')[:-1]
    # Load joint position annotations
    try:
          with pycdf.CDF(path.join(subj_dir, 'Poses_D2_Positions', base_filename + '.cdf')) as cdf:
                poses_2d = np.array(cdf['Pose'])
                poses_2d = poses_2d.reshape(poses_2d.shape[1], 32, 2)
          with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions_mono_universal', base_filename + '.cdf')) as cdf:
                poses_3d_univ = np.array(cdf['Pose'])
                poses_3d_univ = poses_3d_univ.reshape(poses_3d_univ.shape[1], 32, 3)
          with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions_mono', base_filename + '.cdf')) as cdf:
                poses_3d = np.array(cdf['Pose'])
                poses_3d = poses_3d.reshape(poses_3d.shape[1], 32, 3)
          with pycdf.CDF(path.join(subj_dir, 'Poses_D3_Positions', tof_filename + '.cdf')) as cdf:
                poses_3d_original = np.array(cdf['Pose'])
                poses_3d_original = poses_3d_original.reshape(poses_3d_original.shape[1], 32, 3)
          with h5py.File(path.join(subj_dir, 'BBox', base_filename + '.mat'), 'r') as file:
                bboxes = [_process_bbox(file[p[0]].value.T) for p in file['Masks']]
                bboxes = np.array(bboxes)
          with h5py.File(path.join(subj_dir, 'BGSub', base_filename + '.mat'), 'r') as file:
                bgsub = [file[p[0]].value.T for p in file['Masks']]
                bgsub = np.array(bgsub)
 
    except OSError as e:
          print('Error on loading annotations')
          print(path.join(subj_dir, 'Poses_D2_Positions', base_filename + '.cdf')) 
          print(path.join(subj_dir, 'Poses_D3_Positions_mono_universal', base_filename + '.cdf'))
          print(path.join(subj_dir, 'Poses_D3_Positions_mono', base_filename + '.cdf'))
          print(path.join(subj_dir, 'BBox', base_filename + '.mat'))
          print(e.errno, e.strerror, e.filename, e.filename2)
          raise e
    poses_3d_normalized = np.asarray([_make_unit_diagonal(p) for p in poses_3d_univ])
    # Infer camera intrinsics
    camera_int = infer_camera_intrinsics(poses_2d, poses_3d)
    camera_int_univ = infer_camera_intrinsics(poses_2d, poses_3d_univ)

    #frame_indices = np.arange(0,50)
    frame_indices = select_frame_indices_to_include(subject, poses_3d_univ) #UNCOMMENT THIS FOR COMON DATASET******************************************************************
    frames = frame_indices + 1
    video_file = path.join(subj_dir, 'Videos', base_filename + '.mp4')
    frames_dir = path.join(out_dir, 'imageSequence', camera)
    range_dir = path.join(out_dir, 'ToFSequence')
    debug_dir = path.join(out_dir, 'DebugBbox', camera)
    makedirs(frames_dir, exist_ok=True)
    makedirs(range_dir, exist_ok=True)
    makedirs(debug_dir, exist_ok=True)
    scales = np.zeros((bboxes.shape[0],2))
    image_size_multiplier = [1,1]

    # Check to see whether the frame images have already been extracted previously
    existing_files = {f for f in listdir(frames_dir)}
    existing_range_files = {f for f in listdir(range_dir)}
    frames_are_extracted = True
    for i in frames:
        filename = 'img_%06d.jpg' % i
        if filename not in existing_files:
            frames_are_extracted = False
            break
    range_are_extracted = True
    for i in frame_indices:
          filename = 'tof_range%06d.jpg' % i
          if filename not in existing_range_files:
                range_are_extracted = False
                break
    if not frames_are_extracted:
        with TemporaryDirectory() as tmp_dir:
            # Use ffmpeg to extract frames into a temporary directory
            call([
                'ffmpeg',
                '-nostats', '-loglevel', '0',
                '-i', video_file,
                #'-vf', #'-qscale:v', '3', #'scale=224:224',
                '-qscale:v', '3',
                path.join(tmp_dir, 'img_%06d.jpg')
            ])
            print('\nfiles extracted from: %s / to: %s'% ( video_file, tmp_dir))

            # Move included frame images into the output directory
            
            for i in frames:
                idx = i-1
                filename = 'img_%06d.jpg' % i
                img = Image.open(path.join(tmp_dir, filename)) 
                #img = np.array(img) * bgsub[idx][:, :, np.newaxis]
                #img = Image.fromarray(img)
                _transform_image(img, bboxes[idx]).save(path.join(frames_dir, filename)) 


                #move(
                #    path.join(tmp_dir, filename),
                #    path.join(frames_dir, filename)
                #)
                # Applying scales to the 2d coordinates of bounding box and 2d pose
                #scales[idx] = _compute_scales(metadata.sequence_mappings[subject][(camera, '')], [224, 224])
                scales[idx] = _compute_scales(metadata.sequence_mappings[subject][(camera, '')], metadata.sequence_mappings[subject][(camera, '')]) 
                poses_2d[idx] = _transform_2dkp_annots(poses_2d[idx], bboxes[idx]) 

                ''' 
                pose_filename = 'pose_%06d.txt' % i
                pose_norm_filename = 'pose_norm_%06d.txt' % i
                pose_2d_filename = 'pose_2d_%06d.txt' % i
                np.savetxt(path.join(debug_dir, pose_filename),
                           poses_3d_univ[idx])
                np.savetxt(path.join(debug_dir, pose_norm_filename),
                           poses_3d_normalized[idx])
                np.savetxt(path.join(debug_dir, pose_2d_filename),
                           poses_2d[idx])
                img = cv2.imread(path.join(frames_dir, filename))
                cv2.imwrite(path.join(debug_dir, filename), _draw_annot(img, bboxes[idx], poses_2d[idx]))#DEBUG '''
                
    if not range_are_extracted:
        try:
              with pycdf.CDF(path.join(subj_dir, 'TOF', tof_filename + '.cdf')) as cdf:
                    tof_range = _reshape_tof(np.array(cdf['RangeFrames'][0]))
                    tof_int = _reshape_tof(np.array(cdf['IntensityFrames'][0]))
                    tof_sync = np.array(cdf['Index'][0]).astype(int) # 1-indexed array (from matlab)
        except pycdf.CDFError as e:
              print('path joint: ', path.join(subj_dir, 'TOF', tof_filename + '.cdf'))
              print('metadata out: ', metadata.get_base_filename(subject, action, subaction, ''))
              print('subject, aciton, subaction: ', subject, action, subaction)

        for i in frame_indices:
                f_img = _pad_with_zeros(tof_range[tof_sync[i] - 1], (224,224))
                m = np.max(f_img)
                img = cv2.convertScaleAbs(src=f_img, alpha=255/m)
                cv2.imwrite(path.join(range_dir, 'tof_range%06d.jpg' % (i+1)),
                             img)
                cv2.imwrite(path.join(range_dir, 'tof_intensity%06d.jpg' % (i+1)),
                            _decode_tof_intensity(_pad_with_zeros(tof_int[tof_sync[i] - 1], (224,224))))
    return {
        'pose/2d': poses_2d[frame_indices],
        'pose/3d-univ': poses_3d_univ[frame_indices],
        'pose/3d': poses_3d[frame_indices],
        'pose/3d-original': poses_3d_original[frame_indices],
        'intrinsics/' + camera: camera_int,
        'intrinsics-univ/' + camera: camera_int_univ,
        'frame': frames,
        'camera': np.full(frames.shape, int(camera)),
        'subject': np.full(frames.shape, int(included_subjects[subject])),
        'action': np.full(frames.shape, int(action)),
        'subaction': np.full(frames.shape, int(subaction)),
        'bbox': bboxes[frame_indices],
        'pose/3d-norm': poses_3d_normalized[frame_indices]
    }


def process_subaction(subject, action, subaction):
    datasets = {}
    #dir_out = 'processed'
    dir_out = 'video_processed_resized_crop'
    out_dir = path.join(dir_out, subject, metadata.action_names[action] + '-' + subaction)
    makedirs(out_dir, exist_ok=True)

    for camera in tqdm(metadata.camera_ids, ascii=True, leave=False):
        if (subject, action, subaction, camera) in blacklist or (subject, action, subaction, '') in blacklist:
            continue

        try:
            annots = process_view(out_dir, subject, action, subaction, camera)
        except OSError as e:
            print('Error processing sequence, skipping: ', repr((subject, action, subaction, camera)))
            print(e.errno, e.filename, e.strerror)
            raise e
            continue

        for k, v in annots.items():
            if k in datasets:
                datasets[k].append(v)
            else:
                datasets[k] = [v]

    if len(datasets) == 0:
        return

    datasets = {k: np.concatenate(v) for k, v in datasets.items()}

    with h5py.File(path.join(out_dir, 'annot.h5'), 'w') as f:
        for name, data in datasets.items():
            f.create_dataset(name, data=data)


def process_all():
    sequence_mappings = metadata.sequence_mappings

    subactions = []

    for subject in included_subjects.keys():
        subactions += [
            (subject, action, subaction)
            for action, subaction in sequence_mappings[subject].keys()
            if int(action) > 1 and action not in ['54138969', '55011271', '58860488', '60457274']  # Exclude '_ALL'
        ]

    for subject, action, subaction in tqdm(subactions, ascii=True, leave=False):
        process_subaction(subject, action, subaction)


if __name__ == '__main__':
  process_all()
