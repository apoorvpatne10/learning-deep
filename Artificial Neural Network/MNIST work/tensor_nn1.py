#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 19 12:27:56 2018

@author: apoorv
"""
import tensorflow as tf

x = tf.constant(5)
y = tf.constant(6)

#res = tf.multiply(x, y)
print(res)

with tf.Session() as sess:
    print(sess.run(res))
