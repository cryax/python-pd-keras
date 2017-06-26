# -*- coding: utf-8 -*-
import time

import pandas as pd

from younet_rnd_infrastructure.tri.network import network
import pyexcel_xlsx 
import re
from younet_rnd_infrastructure.tri.common import file_tool
import json
import codecs
from pyexcel_xlsx import save_data
from collections import OrderedDict
import pickle

def dump_var_to_file(uvar, filename):    
    with open(filename, 'wb') as f:
        pickle.dump(uvar, f)
       
        
def read_dump_file(filename):
    with open(filename, 'rb') as f:
        tmp = pickle.load(f)        
        return tmp
    
class FindPlace:
    def __init__(self, number_time_retry=2, time_sleep=1):
        self.__PLACES_SERVICE_KEY = "AIzaSyA2fuo6w2ludg_xKtRLAGNvQKFi3SiiAF4"
        self.__GEOCODING_SERVICE_KEY = 'AIzaSyDONShW7LMO_4_6OiKAMSIhlwWKnbtwH4E'
        self.__number_time_retry = number_time_retry
        self.__time_sleep = time_sleep

    def get_full_info_place(self, place_name):
        url = 'https://maps.googleapis.com/maps/api/place/textsearch/json?query=%s&key=%s&type=university&language=en' % (place_name, self.__PLACES_SERVICE_KEY)
#        print url
        number_time_retry = 1
        while number_time_retry < self.__number_time_retry:
            response = network.get(url)
            if 'results' in response.keys():
                results = response['results']

                if len(results) == 0:
#                    print 'Retry: %s-th' % number_time_retry
                    number_time_retry += 1
#                    print 'Going to sleep in %s s' % self.__time_sleep
                    time.sleep(self.__time_sleep)
                    continue
                elif len(results) > 1:
#                    print 'There are more than 1 result. Return the first one'
                    return results[0]
                else:
                    return results[0]

        print 'There is no result'
        return None

    def get_country_name_from_text(self, place_name):
#        print 'Getting country from name: %s' % place_name
        result_json = self.get_full_info_place(place_name)
        if result_json is None:
            return 'Unknown'
        if 'formatted_address' in result_json.keys():
            return result_json['formatted_address'].split(',')[-1]
        else:
            return 'Unknown'
            
    def get_official_school_name_from_alias(self, alias):
#        print 'Getting country from name: %s' % alias
        result_json = self.get_full_info_place(alias)
        if result_json is None:
            return 'Unknown'
        if 'name' in result_json.keys():
            return result_json['name'].split(',')[-1]
        else:
            return 'Unknown'

    def get_country_name_from_coordinate(self, lat, long):
        #2500requests
        url = 'https://maps.googleapis.com/maps/api/geocode/json?latlng=%s,%s&key=%s' % (lat, long, self.__GEOCODING_SERVICE_KEY)
        response = network.get(url)

        results = response['results']
        for result in results:
            address_components = result['address_components']
            for component in address_components:
                address_types = component['types']
                for address_type in address_types:
                    if address_type == 'country':
                        return component['long_name']

        return 'Unknown'

