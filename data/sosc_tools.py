import json
import time
import numpy as np
import itertools
from collections import defaultdict
from PIL import Image, ImageDraw


def _isArrayLike(obj):
    return hasattr(obj, '__iter__') and hasattr(obj, '__len__')


class SOSC:
    def __init__(self, annotation_file=None):
        """SOSC helper class for reading and visualizing annotation"""
        # load dataset
        self.dataset, self.anns, self.cats, self.scenes = dict(), dict(), dict(), dict()
        self.sceToAnns, self.catToScenes = defaultdict(list), defaultdict(list)
        if not annotation_file == None:
            print('loading annotations into memory...')
            tic = time.time()
            dataset = json.load(open(annotation_file, 'r'))
            assert type(dataset) == dict, 'annotation file format {} not supported'.format(type(dataset))
            print('Done (t={:0.2f}s)'.format(time.time() - tic))
            self.dataset = dataset
            # for item in self.dataset['scenes']:
            #     if item['id'] == 3689:
            #         remove_item = item
            #         break
            # self.dataset['scenes'].remove(remove_item)
            self.createIndex()

    def createIndex(self):
        """create index for image, annotations, categories"""
        print('creating index...')
        anns, cats, scenes = {}, {}, {}
        sceToAnns, catToScenes = defaultdict(list), defaultdict(list)
        if 'annotations' in self.dataset:
            for ann in self.dataset['annotations']:
                sceToAnns[ann['scene_id']].append(ann)
                if 'image_id' not in ann:
                    ann['image_id'] = ann['scene_id']
                if 'bbox' not in ann:
                    ann['bbox'] = ann['v_bbox']
                anns[ann['id']] = ann

        if 'scenes' in self.dataset:
            for scene in self.dataset['scenes']:
                scenes[scene['id']] = scene

        if 'categories' in self.dataset:
            for cat in self.dataset['categories']:
                cats[cat['id']] = cat

        if 'annotations' in self.dataset and 'categories' in self.dataset:
            for ann in self.dataset['annotations']:
                catToScenes[ann['category_id']].append(ann['scene_id'])

        print('index created!')

        # create class members
        self.anns = anns
        self.sceToAnns = sceToAnns
        self.catToScenes = catToScenes
        self.scenes = scenes
        self.cats = cats

    def getAnnIds(self, imgIds=[], catIds=[]):
        """
        Get ann ids that satisfy given filter conditions
        :param sceIds: get anns for given scenes
        :param catIds: get anns for given cats
        :return: ids
        """
        sceIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(sceIds) == len(catIds) == 0:
            anns = self.dataset['annotations']
        else:
            if not len(sceIds) == 0:
                lists = [self.sceToAnns[sceId] for sceId in sceIds if sceId in self.sceToAnns]
                anns = list(itertools.chain.from_iterable(lists))
            else:
                anns = self.dataset['annotations']
            anns = anns if len(catIds) == 0 else [ann for ann in anns if ann['category_id'] in catIds]
        ids = [ann['id'] for ann in anns]

        return ids

    def getCatIds(self, catNms=[], supNms=[], catIds=[]):
        """
        Get category ids that satisfy given filter conditions
        :param catNms: get cats for given cat names
        :param supNms: get cats for given supercategory names
        :param catIds: get cats for given cat ids
        :return: integer array of cat ids
        """
        catNms = catNms if _isArrayLike(catNms) else [catNms]
        supNms = supNms if _isArrayLike(supNms) else [supNms]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(catNms) == len(supNms) == len(catIds) == 0:
            # cats = self.dataset['categories']
            cats = self.cats
        else:
            cats = self.dataset['categories']
            cats = cats if len(catNms) == 0 else [cat for cat in cats if cat['name'] in catNms]
            cats = cats if len(supNms) == 0 else [cat for cat in cats if cat['supercategory'] in supNms]
            cats = cats if len(catIds) == 0 else [cat for cat in cats if cat['id'] in catIds]
        ids = [cat['id'] for _, cat in cats.items()]
        ids.sort()
        return ids

    def getImgIds(self, imgIds=[], catIds=[]):
        '''
        Get img ids that satisfy given filter conditions.
        :param sceIds (int array) : get scenes for given ids
        :param catIds (int array) : get scenes with all given cats
        :return: ids (int array)  : integer array of scenes ids
        '''
        sceIds = imgIds if _isArrayLike(imgIds) else [imgIds]
        catIds = catIds if _isArrayLike(catIds) else [catIds]

        if len(sceIds) == len(catIds) == 0:
            ids = self.scenes.keys()
        else:
            ids = set(sceIds)
            for i, catId in enumerate(catIds):
                if i == 0 and len(ids) == 0:
                    ids = set(self.catToScenes[catId])
                else:
                    ids &= set(self.catToScenes[catId])
        return list(ids)

    def loadAnns(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying anns
        :return: anns (object array) : loaded ann objects
        """
        if _isArrayLike(ids):
            return [self.anns[id] for id in ids]
        elif type(ids) == int:
            return [self.anns[ids]]

    def loadCats(self, ids=[]):
        """
        Load cats with the specified ids.
        :param ids (int array)       : integer ids specifying cats
        :return: cats (object array) : loaded cat objects
        """
        if _isArrayLike(ids):
            return [self.cats[id] for id in ids]
        elif type(ids) == int:
            return [self.cats[ids]]

    def loadImgs(self, ids=[]):
        """
        Load anns with the specified ids.
        :param ids (int array)       : integer ids specifying img
        :return: imgs (object array) : loaded img objects
        """
        if _isArrayLike(ids):
            return [self.scenes[id] for id in ids]
        elif type(ids) == int:
            return [self.scenes[ids]]

    def labels2Colors(self, label, palette):
        """Simple function that add fixed colors depending on the class"""
        colors = label * palette
        colors = (colors % 255).astype(np.uint8)

        return colors

    def showAnns(self, image, anns, f_bbox=True, v_bbox=False, f_mask=False, v_mask=True,
                 name=True, layer=True, object_id=False, pair_order=False, dataDir='.'):
        """
        Adds the predicted boxes on the top of the image
        :param image: an image as returned by opencv
        :param anns: annotations contains of full bbox, visible bbox
        :param show_f: flag to show the full bbox or not
        :param show_v: flag to show the full bbox or not
        :return: overlay image
        """
        if len(anns) == 0:
            return image
        # used to make colors for each class
        palette = np.array([2 ** 11 - 1, 2 ** 21 - 1, 2 ** 31 - 1])
        # image size
        for ann in anns:
            label = ann['category_id']
            color = tuple(self.labels2Colors(label, palette))
            if f_mask:
                image = np.array(image)
                rgba = Image.open('%s/%s'%(dataDir, ann['f_img_name']))
                r, g, b, a = rgba.split()
                for c in range(3):
                    image[:, :, c] = np.where(np.array(a) > 0, image[:, :, c] * 0.5 + color[c] * 0.5, image[:, :, c])
                image = Image.fromarray(image)
            if v_mask:
                image = np.array(image)
                v = Image.open('%s/%s' % (dataDir, ann['v_mask_name']))
                for c in range(3):
                    image[:, :, c] = np.where(np.array(v) > 0, image[:, :, c] * 0.5 + (color[c]+50).astype(np.uint8) * 0.5, image[:, :, c])
                image = Image.fromarray(image)
            d = ImageDraw.Draw(image)
            if f_bbox:
                f_box = ann['f_bbox']
                d.rectangle((f_box[0], f_box[1], f_box[0]+f_box[2], f_box[1]+f_box[3]), outline=color)
            if v_bbox:
                v_box = ann['v_bbox']
                d.rectangle((v_box[0], v_box[1], v_box[0]+v_box[2], v_box[1]+v_box[3]), outline=color)
            if name:
                v_box = ann['v_bbox']
                class_name = self.cats[ann['category_id']]['supercategory']
                d.text((v_box[0], v_box[1]), class_name, fill=(255, 255, 255))
            if layer:
                v_box = ann['v_bbox']
                layer_name = str(ann['layer_order'])
                d.text((v_box[0]+v_box[2]-10, v_box[1]+v_box[3]-10), layer_name, fill=(255, 255, 255))
            if object_id:
                v_box = ann['v_bbox']
                class_name = self.cats[ann['category_id']]['supercategory']
                object_name = str(ann['id'])
                d.text((v_box[0] + len(class_name)*6+2, v_box[1]), object_name, fill=(255, 255, 255))
            if pair_order:
                print(ann['id'], ann['pair_order'])

        return image