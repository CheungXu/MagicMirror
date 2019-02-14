import DataCooker as DC
import cv2

def Label_Read_Test():
    data = DC.DataCooker()
    print('Dictory:')
    print(data.show_label_dict())
    print('List:')
    print(data.show_image_names())

def Image_batch_Test():
    data = DC.DataCooker(batch_size=4)
    images, labels = data.get_batch(0)
    for i in range(3):
        cv2.imshow('1',images[i])
        cv2.waitKey(0)
        cv2.destroyAllWindows()

def get_available_gpus():
    """
    code from http://stackoverflow.com/questions/38559755/how-to-get-current-available-gpus-in-tensorflow
    """
    from tensorflow.python.client import device_lib as _device_lib
    local_device_protos = _device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']

if __name__ == '__main__':
    Image_batch_Test()
    #res = get_available_gpus()
    #print(res)