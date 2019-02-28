#!/usr/bin/env python3

import xml.etree.ElementTree as ET
import numpy as np

def _parse_matrix(string):
      rows_split = string[1:-1].split(";")
      numbers = np.asarray([p.split(' ') for p in rows_split]).ravel()
      array = np.asarray([int(x) for x in numbers])
      return array.reshape(-1, 2)

class H36M_Metadata:
    def __init__(self, metadata_file):
        self.subjects = []
        self.sequence_mappings = {}
        self.action_names = {}
        self.camera_ids = []
        self.camera_resolutions = {}
        subject_from_xml = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9', 'S10', 'S11']
        tree = ET.parse(metadata_file)
        root = tree.getroot()
        self.root_ = root
        self.camera_ids = [elem.text for elem in root.find('dbcameras/index2id')]
        for i, tr in enumerate(root.find('mapping')):
            a,b, *args = [td.text for td in tr]
            if i == 0:
                _, _, *self.subjects = [td.text for td in tr]
                self.sequence_mappings = {subject: {} for subject in self.subjects}
            elif i < 33:
                action_id, subaction_id, *prefixes = [td.text for td in tr]
                for subject, prefix in zip(self.subjects, prefixes):
                    self.sequence_mappings[subject][(action_id, subaction_id)] = prefix
            if a == b == None:
                  for i,e in enumerate(args):
                        res_matrix = _parse_matrix(e)
                        for k,res in enumerate(res_matrix):
                              self.sequence_mappings[subject_from_xml[i]][(self.camera_ids[k], '')] = res

        for i, elem in enumerate(root.find('actionnames')):
            action_id = str(i + 1)
            self.action_names[action_id] = elem.text

        

    def get_base_filename(self, subject, action, subaction, camera):
        return '{}.{}'.format(self.sequence_mappings[subject][(action, subaction)], camera)


def load_h36m_metadata():
    return H36M_Metadata('metadata.xml')


if __name__ == '__main__':
    metadata = load_h36m_metadata()
    print(metadata.subjects)
    print(metadata.sequence_mappings)
    print(metadata.action_names)
    print(metadata.camera_ids)
