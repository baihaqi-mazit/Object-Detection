# coding=utf-8
import cv2
import random
import numpy as np
import torch

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            # img = np.fliplr(img)
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
        return img, bboxes


class RandomCrop(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            img_copy = img.copy()
            bboxes_copy = bboxes

            try:
                max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            except ValueError:
                print('bboxes[:, 0:2]', bboxes[:, 0:2])
                print('bboxes[:, 2:4]', bboxes[:, 2:4])
                print('img_copy', img_copy.shape, 'img', img.shape)
                print('bboxes_copy', type(bboxes_copy), 'bboxes', type(bboxes))
                print('bboxes_copy', bboxes_copy.size, 'bboxes', bboxes.size)
                return img_copy, bboxes_copy
                exit()
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = max(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))
            crop_ymax = max(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))

            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] - crop_xmin
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] - crop_ymin

            # print('len(bboxes.tolist())', len(bboxes.tolist()))
            # if len(bboxes.tolist()) == 0:
            #     img = img_copy
            #     bboxes = bboxes_copy
        return img, bboxes


class RandomAffine(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            img_copy = img.copy()
            bboxes_copy = bboxes
            
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] + tx
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] + ty

            # if len(bboxes.tolist()) == 0:
            #     img = img_copy
            #     bboxes = bboxes_copy
        return img, bboxes


class Resize(object):
    """
    Resize the image to target size and transforms it into a color channel(BGR->RGB),
    as well as pixel value normalization([0,1])
    """
    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape

        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        try:
            image_resized = cv2.resize(img, (resize_w, resize_h))
        except:
            print(resize_w, resize_h, resize_ratio, w_org, h_org)

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0  # normalize to [0, 1]

        if self.correct_box:
            bboxes[:, [0, 2]] = bboxes[:, [0, 2]] * resize_ratio + dw
            bboxes[:, [1, 3]] = bboxes[:, [1, 3]] * resize_ratio + dh
            return image, bboxes
        return image


class Mixup(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        return img, bboxes


class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes


# Modify from Hou Zhi Ch'ng
class RandomChangeColor(object):
    def __init__(self, p=0.2):
        self.p = p
        self.color_list = [
            [228,238,240], [148,236,255], [174,174,255],
            [103,255,220], [124,229,176], [231,216,180]
        ]

    def __call__(self, img, bboxes):
        if random.random() > self.p:
            random_color = random.choice(self.color_list)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)
            img = img.copy()
            img[mask == 0] = random_color
        return img, bboxes


class Dilation(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() > self.p:
            kernel = np.ones((2,2),np.uint8)
            _, mask = cv2.threshold(img,220,255,cv2.THRESH_BINARY_INV)
            dst = cv2.dilate(mask, kernel, iterations = 1)
            img = cv2.bitwise_not(dst)
        return img, bboxes


class Smudge(object):
    def __init__(self, p=0.2):
        self.p = p
        self._dist_transform_algo = {
            'b': cv2.DIST_L2,
            'g': cv2.DIST_L1,
            'r': cv2.DIST_C
        }
    
    def __call__(self, img, bboxes):
        b, g, r = cv2.split(img)

        b = self.transform(b, 'b')
        g = self.transform(g, 'g')
        r = self.transform(r, 'r')

        dist = cv2.merge((b, g, r))
        dist = cv2.normalize(dist, dist, 0, 4.0, cv2.NORM_MINMAX)
        dist = np.dot(dist.astype('float64'), [0.299, 0.587, 0.114])
        dist = np.moveaxis(np.stack([dist, dist, dist]), 0, -1)

        data = dist / 4.0
        data = 1800 * data
        smudged_img = data.astype('uint16')

        return smudged_img, bboxes
    
    def basic_transform(self, img):
        _, mask = cv2.threshold(img, 220, 255, cv2.THRESH_BINARY_INV)
        img = cv2.bitwise_not(mask)
        return img

    def transform(self, channel, _type):
        channel = self.basic_transform(channel)
        channel = cv2.distanceTransform(channel, self._dist_transform_algo[_type], 5)
        channel = cv2.normalize(channel, channel, 0, 1.0, cv2.NORM_MINMAX)
        return 
        

class Blur(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() > self.p:
            img = cv2.GaussianBlur(img, (5,5), 0)
        return img, bboxes


class Brightness(object):
    def __init__(self, p=0.3):
        self.p = p
        
    def __call__(self, img, bboxes):
        alpha = np.array([50.])
        if random.random() > self.p:
            if random.random() > 0.5:
                img = cv2.add(img, alpha)
            else:
                img = cv2.subtract(img, alpha)
        return img, bboxes


# Source: https://github.com/Paperspace/DataAugmentationForObjectDetection
class RandomRotate(object):
    """Randomly rotates an image    
    
    
    Bounding boxes which have an area of less than 25% in the remaining in the 
    transformed image is dropped. The resolution is maintained, and the remaining
    area if any is filled by black color.
    
    Parameters
    ----------
    angle: float or tuple(float)
        if **float**, the image is rotated by a factor drawn 
        randomly from a range (-`angle`, `angle`). If **tuple**,
        the `angle` is drawn randomly from values specified by the 
        tuple
        
    Returns
    -------
    
    numpy.ndaaray
        Rotated image in the numpy format of shape `HxWxC`
    
    numpy.ndarray
        Tranformed bounding box co-ordinates of the format `n x 4` where n is 
        number of bounding boxes and 4 represents `x1,y1,x2,y2` of the box
        
    """

    def __init__(self, angle = 10):
        self.angle = angle
        
        if type(self.angle) == tuple:
            assert len(self.angle) == 2, "Invalid range"  
            
        else:
            self.angle = (-self.angle, self.angle)
            
    def __call__(self, img, bboxes):
        angle = random.uniform(*self.angle)
    
        w,h = img.shape[1], img.shape[0]
        cx, cy = w//2, h//2
    
        img = self.rotate_im(img, angle)
    
        corners = self.get_corners(bboxes)
    
        corners = np.hstack((corners, bboxes[:,4:]))
    
    
        corners[:,:8] = self.rotate_box(corners[:,:8], angle, cx, cy, h, w)
    
        new_bbox = self.get_enclosing_box(corners)
    
    
        scale_factor_x = img.shape[1] / w
    
        scale_factor_y = img.shape[0] / h
    
        img = cv2.resize(img, (w,h))
    
        new_bbox[:,:4] /= [scale_factor_x, scale_factor_y, scale_factor_x, scale_factor_y] 
    
        bboxes  = new_bbox
    
        bboxes = self.clip_box(bboxes, [0,0,w, h], 0.25)
    
        return img, bboxes
    
    def rotate_im(self, image, angle):
        """Rotate the image.
        
        Rotate the image such that the rotated image is enclosed inside the tightest
        rectangle. The area not occupied by the pixels of the original image is colored
        black. 
        
        Parameters
        ----------
        
        image : numpy.ndarray
            numpy image
        
        angle : float
            angle by which the image is to be rotated
        
        Returns
        -------
        
        numpy.ndarray
            Rotated Image
        
        """
        # grab the dimensions of the image and then determine the
        # centre
        (h, w) = image.shape[:2]
        (cX, cY) = (w // 2, h // 2)

        # grab the rotation matrix (applying the negative of the
        # angle to rotate clockwise), then grab the sine and cosine
        # (i.e., the rotation components of the matrix)
        M = cv2.getRotationMatrix2D((cX, cY), angle, 1.0)
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])

        # compute the new bounding dimensions of the image
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))

        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cX
        M[1, 2] += (nH / 2) - cY

        # perform the actual rotation and return the image
        image = cv2.warpAffine(image, M, (nW, nH), borderMode=cv2.BORDER_CONSTANT, borderValue=(255,255,255))

        return image
    
    def get_enclosing_box(self, corners):
        """Get an enclosing box for ratated corners of a bounding box
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`  
        
        Returns 
        -------
        
        numpy.ndarray
            Numpy array containing enclosing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
            
        """
        x_ = corners[:,[0,2,4,6]]
        y_ = corners[:,[1,3,5,7]]
        
        xmin = np.min(x_,1).reshape(-1,1)
        ymin = np.min(y_,1).reshape(-1,1)
        xmax = np.max(x_,1).reshape(-1,1)
        ymax = np.max(y_,1).reshape(-1,1)
        
        final = np.hstack((xmin, ymin, xmax, ymax,corners[:,8:]))
        
        return final

    def rotate_box(self, corners,angle,  cx, cy, h, w):
        """Rotate the bounding box.
        
        Parameters
        ----------
        
        corners : numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        
        angle : float
            angle by which the image is to be rotated
            
        cx : int
            x coordinate of the center of image (about which the box will be rotated)
            
        cy : int
            y coordinate of the center of image (about which the box will be rotated)
            
        h : int 
            height of the image
            
        w : int 
            width of the image
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N rotated bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`
        """

        corners = corners.reshape(-1,2)
        corners = np.hstack((corners, np.ones((corners.shape[0],1), dtype = type(corners[0][0]))))
        
        M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
        
        
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # adjust the rotation matrix to take into account translation
        M[0, 2] += (nW / 2) - cx
        M[1, 2] += (nH / 2) - cy
        # Prepare the vector to be transformed
        calculated = np.dot(M,corners.T).T
        
        calculated = calculated.reshape(-1,8)
        
        return calculated
    
    def get_corners(self, bboxes):
        """Get corners of bounding boxes
        
        Parameters
        ----------
        
        bboxes: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        returns
        -------
        
        numpy.ndarray
            Numpy array of shape `N x 8` containing N bounding boxes each described by their 
            corner co-ordinates `x1 y1 x2 y2 x3 y3 x4 y4`      
            
        """
        width = (bboxes[:,2] - bboxes[:,0]).reshape(-1,1)
        height = (bboxes[:,3] - bboxes[:,1]).reshape(-1,1)
        
        x1 = bboxes[:,0].reshape(-1,1)
        y1 = bboxes[:,1].reshape(-1,1)
        
        x2 = x1 + width
        y2 = y1 
        
        x3 = x1
        y3 = y1 + height
        
        x4 = bboxes[:,2].reshape(-1,1)
        y4 = bboxes[:,3].reshape(-1,1)
        
        corners = np.hstack((x1,y1,x2,y2,x3,y3,x4,y4))
        
        return corners

    def bbox_area(self, bbox):
        return (bbox[:,2] - bbox[:,0])*(bbox[:,3] - bbox[:,1])

    def clip_box(self, bbox, clip_box, alpha):
        """Clip the bounding boxes to the borders of an image
        
        Parameters
        ----------
        
        bbox: numpy.ndarray
            Numpy array containing bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes and the bounding boxes are represented in the
            format `x1 y1 x2 y2`
        
        clip_box: numpy.ndarray
            An array of shape (4,) specifying the diagonal co-ordinates of the image
            The coordinates are represented in the format `x1 y1 x2 y2`
            
        alpha: float
            If the fraction of a bounding box left in the image after being clipped is 
            less than `alpha` the bounding box is dropped. 
        
        Returns
        -------
        
        numpy.ndarray
            Numpy array containing **clipped** bounding boxes of shape `N X 4` where N is the 
            number of bounding boxes left are being clipped and the bounding boxes are represented in the
            format `x1 y1 x2 y2` 
        
        """
        ar_ = (self.bbox_area(bbox))
        x_min = np.maximum(bbox[:,0], clip_box[0]).reshape(-1,1)
        y_min = np.maximum(bbox[:,1], clip_box[1]).reshape(-1,1)
        x_max = np.minimum(bbox[:,2], clip_box[2]).reshape(-1,1)
        y_max = np.minimum(bbox[:,3], clip_box[3]).reshape(-1,1)
        
        bbox = np.hstack((x_min, y_min, x_max, y_max, bbox[:,4:]))
        
        delta_area = ((ar_ - self.bbox_area(bbox))/ar_)
        
        mask = (delta_area < (1 - alpha)).astype(int)
        
        bbox = bbox[mask == 1,:]


        return bbox

def display_image(image, winname='Image'):
    cv2.imshow(winname, image)
    cv2.waitKey(-1)
    cv2.destroyAllWindows()