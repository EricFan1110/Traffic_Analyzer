import requests
from bs4 import BeautifulSoup
import folium
import webbrowser
import cv2
import wget
from threading import Thread
from time import sleep
import os
import yolov5
from folium.plugins import HeatMap
import torch
from datetime import datetime


def normalize(val, min_val, max_val):
    return (val - min_val) / (max_val - min_val)


def camera_control(intersection, output, model):
    img_counter = 0
    id = intersection[1]
    coord = intersection[2]
    link = "https://www.richmond.ca/services/ttp/trafficcamerasmap/IntersectionDetails.aspx?id=" + str(id)
    data = []
    html = requests.get(link).text
    soup = BeautifulSoup(html, 'html.parser')
    for i in soup.find_all('b'):
        cam_dir = i.getText()
        data.append(cam_dir)

    while True:
        html = requests.get(link).text
        soup = BeautifulSoup(html, 'html.parser')
        dt = datetime.now()
        index = 0
        predicted_data = [coord]

        try:
            for i in soup.find_all('img'):
                img_link = str(i.get('src'))

                if img_link == "offline.jpg":
                    predicted_data.append([data[index], "offline"])
                    index += 1

                elif img_link.startswith("https://www.richmond.ca/trafficcam/"):
                    filename = wget.download(img_link)
                    np_image = cv2.imread(filename)
                    os.remove(filename)
                    np_image = cv2.resize(np_image, (640, 640))
                    results = model(np_image)

                    # parse results
                    predictions = results.pred[0]
                    boxes = predictions[:, :4]  # x1, y1, x2, y2
                    number_of_predicted_cars = list(predictions.shape)[0]

                    for i in range(number_of_predicted_cars):
                        a, b, c, d = boxes[i]
                        color = (255, 0, 0)
                        cv2.rectangle(np_image, (int(a.item()), int(b.item())), (int(c.item()), int(d.item())), color)

                    # cv2.imwrite(r'saved_Images\{}_{}_{}'.format(id, index, img_counter) + ".png", np_image)
                    predicted_data.append([data[index], number_of_predicted_cars])
                    index += 1
            output.append(predicted_data)

            img_counter += 1
            sleep(120 - (datetime.now() - dt).total_seconds())
        except:
            print("Error has occured")
            predicted_data = [coord]
            for cam in data:
                predicted_data.append([cam, "offline"])
            output.append(predicted_data)
            img_counter += 1
            sleep(120 - (datetime.now() - dt).total_seconds())


if __name__ == '__main__':
    intersections = []

    # Load all data of available surveillance cameras into the intersections list
    # region
    web_url = 'https://www.richmond.ca/services/ttp/trafficcamerasmap/Default.aspx'
    html_text = requests.get(web_url).text
    soup = BeautifulSoup(html_text, 'html.parser')

    all_data = soup.find_all("script")

    for i in range(13, 147):
        camera = all_data[i].text.splitlines()
        for line_index in range(len(camera)):
            camera[line_index] = camera[line_index].strip()

        coord_str = camera[1].removeprefix("L.marker([").removesuffix("], {").replace(" ", "")
        coord_str = [coord_str.split(sep=",")]
        coord = [float(str(coord_str[0][0])), float(str(coord_str[0][1]))]

        name = camera[6].removeprefix("availableTags.push(\'").removesuffix("\');")

        camera_id = int(camera[5].removeprefix(".bindPopup(\"<div class=\'subtitle\'>" + name +
                                               "</div><a href=\'https://www.richmond.ca/services/ttp"
                                               "/trafficcamerasmap/IntersectionDetails.aspx?id=").removesuffix(
            "\'>Live Traffic Camera Images</a>\");"))

        intersections.append((name, camera_id, coord))
    # endregion
    print(intersections)

    mymap = folium.Map(
        location=[49.1666, -123.1336],
        zoom_start=13
    )

    for intersection in intersections:
        coord = intersection[2]
        folium.Marker(
            location=[coord[0], coord[1]]
        ).add_to(mymap)

    mymap.save('map.html')

    model = torch.hub.load('ultralytics/yolov5', 'yolov5l')
    model.classes = [2]
    predicted_datas = []
    for intersection in intersections:
        t = Thread(target=camera_control, args=(intersection, predicted_datas, model))
        t.start()

    dt = datetime.now()

    img_counter = 0
    while True:
        if len(predicted_datas) == len(intersections):
            max_car = 0
            min_car = float('inf')
            heatmap_data = []

            for predicted_data in predicted_datas:
                intersection_car = 0
                for i in range(len(predicted_data)):
                    if i != 0:
                        if predicted_data[i][1] == 'offline':
                            continue  # This may make min_car value incorrect because intersection_car is still being
                            # compared to min_car even if camera is offline
                        intersection_car += predicted_data[i][1]
                if intersection_car > max_car:
                    max_car = intersection_car
                if intersection_car < min_car:
                    min_car = intersection_car

            for predicted_data in predicted_datas:
                intersection_car = 0
                for i in range(len(predicted_data)):
                    if i != 0:
                        if predicted_data[i][1] == 'offline':
                            continue
                        intersection_car += predicted_data[i][1]

                data = predicted_data[0]
                data.append(normalize(intersection_car, min_car, max_car))
                heatmap_data.append(data)

            HeatMap(heatmap_data).add_to(mymap)
            mymap.save('map{}.html'.format(img_counter))
            with open('saved_Data.txt', 'a') as f:
                f.write(str(heatmap_data) + ',\n')

            for i in range(len(predicted_datas)):
                predicted_datas.pop()

            img_counter += 1
