#!/usr/bin/env python
# -*- coding: utf-8 -*-
# vim:fenc=utf-8
#
# Copyright © 2019 Theo Morales <theo.morales.fr@gmail.com>
#
# Distributed under terms of the MIT license.


'''
Utility script to extract frames from the given topic
(usually used with rosbag).
'''

import os
import sys
import cv2
import rospy
import cv_bridge
import message_filters

from tqdm import tqdm
from collections import OrderedDict
from geometry_msgs.msg import TransformStamped
from sensor_msgs.msg import CompressedImage, Image

class FrameExtractor():
    def __init__(self):
        self.output_path = rospy.get_param('~output', 'extracted_output/')
        self.raw = rospy.get_param('~raw', False)
        self.img_topic = rospy.get_param('~images', None)
        self.images = OrderedDict()
        self.first_second = None

        if self.img_topic is None:
            rospy.logwarn("""FrameExtractor: rosparam '~images' has not been specified!
Typical command-line usage:
      $ ./extract_frames _images:=<image_topic> _output:=<output_path> _raw:=<1|0>'""")
            sys.exit(1)

    def extract(self):
        if not os.path.isdir(self.output_path):
            os.mkdir(self.output_path)
        if self.raw:
            imgs_sub = [message_filters.Subscriber(self.img_topic, Image)]
        else:
            imgs_sub = [message_filters.Subscriber(self.img_topic, CompressedImage)]

        imgs_sync = message_filters.TimeSynchronizer(
            imgs_sub, queue_size=1000)

        if self.raw:
            imgs_sync.registerCallback(self._save_images)
        else:
            imgs_sync.registerCallback(self._save_compressed_images)

        print("[*] Extracting...")
        raw_input("[*] Press Enter when the stream is complete...")
        sys.exit(0)

    def _save_compressed_images(self, *img_messages):
        for i, img_msg in enumerate(img_messages):
            if not self.first_second:
                self.first_second = img_msg.header.stamp.secs
            seconds = img_msg.header.stamp.secs - self.first_second
            nanoseconds = img_msg.header.stamp.nsecs
            fname = "%04d_%09d.jpg" % (seconds, nanoseconds)
            self.images[fname] = (1000000000 * seconds) + nanoseconds
            with open(os.path.join(self.output_path, fname), 'w') as img_file:
                img_file.write(img_msg.data)

    def _save_raw_images(self, *img_messages):
        bridge = cv_bridge.CvBridge()
        for i, img_msg in enumerate(img_messages):
            if not self.first_second:
                self.first_second = img_msg.header.stamp.secs
            img = bridge.imgmsg_to_cv2(img_msg, desired_encoding="8UC3")
            channels = img.shape[2] if img.ndim == 3 else 1
            encoding_in = bridge.dtype_with_channels_to_cvtype2(img.dtype, channels)
            img = cv_bridge.cvtColorForDisplay(
                img, encoding_in="rgb8", encoding_out='',
                do_dynamic_scaling=self.do_dynamic_scaling)
            seconds = img_msg.header.stamp.secs - self.first_second
            nanoseconds = img_msg.header.stamp.nsecs
            fname = "%04d_%09d.jpg" % (seconds, nanoseconds)
            self.images[fname] = (1000000000 * seconds) + nanoseconds
            cv2.imwrite(fname, img)



if __name__ == '__main__':
    rospy.init_node('extract_frames')
    extractor = FrameExtractor()
    extractor.extract()
    rospy.spin()
