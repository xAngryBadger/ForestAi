import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET

def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        filename = root.find('filename').text
        for member in root.findall('object'):
            xmin = int(member.find('bndbox/xmin').text)
            ymin = int(member.find('bndbox/ymin').text)
            xmax = int(member.find('bndbox/xmax').text)
            ymax = int(member.find('bndbox/ymax').text)
            label = member.find('name').text
            value = (filename, xmin, ymin, xmax, ymax, label)
            xml_list.append(value)
    column_name = ['filename', 'xmin', 'ymin', 'xmax', 'ymax', 'Tree']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df

def main():
    image_path = os.path.join(os.getcwd(), 'C:\HololeucaGPU\hololeuca img\fedegoso') # Path to the directory containing the XML files
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('labels.csv', index=False) # Saves the dataframe to labels.csv
    print('Successfully converted xml to csv.')

if __name__ == '__main__':
    main()