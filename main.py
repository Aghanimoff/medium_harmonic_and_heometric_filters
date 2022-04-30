from tkinter.filedialog import askopenfilename, asksaveasfilename
from tkinter import *
import tkinter
from tkinter.colorchooser import askcolor

import cv2
from cv2 import imwrite
import matplotlib.pyplot as plt
import numpy as np

import time


def rgb_2_gray(r, g, b):
    """
    Функция возвращает серый оттенок из RGB

    Формула яркости повзаимствованна из алгоритма преобразования RGB - YIQ
    она же - формула NTSC для получения серых оттенков из RGB
    """
    return round(0.299*r + 0.587*g + 0.114*b)


def open_filtered_image():
    def show_color(event, x, y, flags, param):
        """
        Функция для передачи значения цвета пикселя на указателе мышки
        в окне cv2 в соответствующий tkinter блок

        Использует глобальные переменные
        """

        if is_gray:
            _b = _g = _r = image[y, x].astype(int)
            r_16 = g_16 = b_16 = format(_r, '02x')

            B = G = R = new_image[y, x].astype(int)
            R_16 = G_16 = B_16 = format(R, '02x')

            old_bright = _r
            new_bright = R

        else:
            _b, _g, _r = image[y, x].astype(int)
            r_16, g_16, b_16 = format(_r, '02x'), format(_g, '02x'), format(_b, '02x')

            B, G, R = new_image[y, x].astype(int)
            R_16, G_16, B_16 = format(R, '02x'), format(G, '02x'), format(B, '02x')

            old_bright = rgb_2_gray(_r, _g, _b)
            new_bright = rgb_2_gray(R, G, B)

        color_before_value['text'] = f'{_r} {_g} {_b}'
        color_before_value['bg'] = f'#{r_16}{g_16}{b_16}'
        color_before_value['fg'] = '#ffffff' if old_bright < 127 else '#000000'

        color_after_value['text'] = f'{R} {G} {B}'
        color_after_value['bg'] = f'#{R_16}{G_16}{B_16}'
        color_after_value['fg'] = '#ffffff' if new_bright < 127 else '#000000'

    def medium_harmonic_func(normalize_k, _array):
        """
        Формула среднегармонического фильтра
        Возвращает значение пикселя

        :param normalize_k: int
        :param _array: np.array
        :return: int
        """
        return normalize_k / np.sum(np.divide(1, _array))

    def medium_geometric_func(normalize_k, _array):
        """
        Формула среднегеометрического фильтра
        Возвращает значение пикселя

        :param normalize_k: int
        :param _array: np.array
        :return: int
        """
        return np.prod(np.power(_array, 1/normalize_k))

    def calculate_mask_to_array():
        """
        Алгоритм расчета матрицы
        Возвращает обработанную матрицу
        :return: np.array

        Использует глобальные переменные
        """
        rows, cols = color.shape[:2]
        array = np.zeros((rows, cols),)  # создаём мустую матрицу

        border_color = int(border_constant_color_button['bg'][(e * 2 + 1):(e * 2 + 3)], 16)
        # Изменяем основное изображение, расширяя границы в соответствии с пользовательскими настройками
        pad_img = cv2.copyMakeBorder(src=color, top=radius, bottom=radius, left=radius, right=radius,
                                     borderType=border_type_dict[border_type_var.get()],
                                     value=border_color if border_type_dict[border_type_var.get()] == cv2.BORDER_CONSTANT else None)

        # проход по каждому пикселю (кроме границ)
        for row in range(radius, rows + radius):
            for col in range(radius, cols + radius):
                # матрица вокруг текущего пикселя в соответствии с размером апертуры
                filter_array = pad_img[(row - r_rad):(row + r_rad + 1), (col - c_rad):(col + c_rad + 1)]
                # вызов функции выбранного фильтра и запись значения пикселя в матрицу нового изображения
                array[row - radius, col - radius] = selected_filter_function(normalize, filter_array)
        return array

    def convert_border_color_to_gray():
        """
        Функция изменения цвета границ константы с RGB на Gray

        Использует глобальные переменные
        """
        border_bgr = hex_to_10_color(border_constant_color_button['bg'])
        gray_color = rgb_2_gray(*border_bgr[::-1])
        border_constant_color_button['bg'] = '#' + format(gray_color, '02x')*3

    mask = aperture.get()
    radius = mask // 2

    c_rad, r_rad = radius, radius
    normalize = mask ** 2
    if dimensional.get() == 'двумерный':
        pass
    elif dimensional.get() == 'одномерный горизонтальный':
        r_rad = 0
        normalize = mask
    elif dimensional.get() == 'одномерный вертикальный':
        c_rad = 0
        normalize = mask

    selected_filter_function = medium_harmonic_func
    if selected_filter.get() == 'среднегармонический':
        pass
    elif selected_filter.get() == 'среднегеометрический':
        selected_filter_function = medium_geometric_func

    filter_start_time = time.time()
    start_time = time.time()

    is_gray = False

    def hex_to_10_color(string):
        # преобразование формата записи цвета из #rrggbb в (B, G, R)
        return tuple(int(string[1:][i:i + 2], 16) for i in (4, 2, 0))

    if gray_image_dict[gray_image.get()] == 'yuv':
        image = cv2.imread(original_image_path_field.get())
        yuv_image = cv2.cvtColor(image, cv2.COLOR_BGR2YUV)
        yuv_image.astype(float)
        new_image = np.zeros_like(yuv_image)
        new_image[:, :, 1:] = yuv_image[:, :, 1:]
        e = 0
        color = yuv_image[:, :, 0]
        new_image[:, :, 0] = calculate_mask_to_array()
        new_image = cv2.cvtColor(new_image, cv2.COLOR_YUV2BGR)

    elif gray_image_dict[gray_image.get()] == 'gray':
        is_gray = True
        convert_border_color_to_gray()
        image = cv2.imread(original_image_path_field.get(), cv2.IMREAD_GRAYSCALE)
        image.astype(float)
        new_image = np.zeros_like(image)
        e = 0
        color = image
        new_image = calculate_mask_to_array()

    elif gray_image_dict[gray_image.get()] == 'rgb':
        image = cv2.imread(original_image_path_field.get())
        image.astype(float)
        b, g, r = cv2.split(image)
        new_image = np.zeros_like(image)
        for e, color in enumerate((b, g, r),):
            new_image[:, :, e] = calculate_mask_to_array()

    new_image = np.uint8(new_image)

    total_time['text'] = f'общее время: {round(time.time() - start_time, 5)}'
    filter_time['text'] = f'время алгоритма: {round(time.time() - filter_start_time, 5)}'

    global new_image_var
    new_image_var = new_image

    global old_image_var
    old_image_var = image

    cv2.namedWindow('main_image')
    cv2.setMouseCallback('main_image', show_color)
    cv2.namedWindow('new_image')
    cv2.setMouseCallback('new_image', show_color)

    cv2.imshow('new_image', new_image)
    cv2.imshow('main_image', image)


def find_image():
    file_path = askopenfilename(filetypes=(('image files', '.png .jpg'), ('All Files', '*.*')))
    if file_path:
        original_image_path_field.delete(0, END)
        original_image_path_field.insert(0, file_path)

        original_image_path_field.xview_moveto(1)  # Если путь файла больше поля - показывать конец


def show_image():
    new_frame = Toplevel()
    image = PhotoImage(file=original_image_path_field.get())  # image not visual
    canvas = Canvas(new_frame, width=image.width(), height=image.height())
    canvas.pack(expand=YES, fill=BOTH)
    canvas.create_image(0, 0, image=image, anchor=NW)  # assigned the gif1 to the canvas object
    canvas.gif1 = image


def save_filtered_image():
    file_name = asksaveasfilename(defaultextension='.png', filetypes=(('PNG', '.png'),))
    if file_name:
        imwrite(file_name, new_image_var)


def histogram_open(img):
    """
    Открывает окно с трехцветной или одноцветной гистограммой

    :param img: image var
    Использует глобальные переменные:
    """
    fig = plt.figure(figsize=(8, 4))
    ax = fig.add_subplot(121)

    if len(img.shape) == 3:  # трехцветный
        b, g, r = cv2.split(img)
        ax.imshow(img[..., ::-1])

        ax = fig.add_subplot(122)
        for x, c in zip([b, g, r], ["b", "g", "r"]):
            xs = np.arange(256)
            ys = cv2.calcHist([x], [0], None, [256], [0, 256])
            ax.plot(xs, ys.ravel(), color=c)

    else:
        ax.imshow(np.repeat(img[:, :, np.newaxis], 3, axis=2))
        ax = fig.add_subplot(122)

        xs = np.arange(256)
        ys = cv2.calcHist([img], [0], None, [256], [0, 256])
        ax.plot(xs, ys.ravel())

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    plt.show()


def original_histogram_open():
    histogram_open(old_image_var)


def filtered_histogram_open():
    histogram_open(new_image_var)


main_frame = Tk()
main_frame.title('среднегармонический и среднегеометрический фильтры')


original_image_path_field = Entry(main_frame, font=40, width=55)
find_image_button = Button(main_frame, text='Открыть', command=find_image, font=40)
show_image_button = Button(main_frame, text='Показать', command=show_image, font=40)

old_image_var = None
new_image_var = None
open_filtered_image_button = Button(main_frame, text='Применить', command=open_filtered_image, font=40)
save_filtered_image_button = Button(main_frame, text='Сохранить', command=save_filtered_image, font=40)


# =========================
def onScale(value):
    value = int(value)
    # у tkinter есть проблемы с шагом для Scale, он не умеет делить шкалу на нечётные значения (1,3,5..)
    # поэтому выкручиваемся и насильно запрещаем устанавливать чётное значение, для пользователя это незаметно
    if not value % 2:
        value = value+1
    aperture_scale.set(value)
    aperture.set(value)


aperture = IntVar()  # хранит значение scale
aperture.set(3)
aperture_scale = Scale(from_=3, to_=21, command=onScale, orient='horizontal')

# =========================


color_before_label = Label(main_frame, font=40, text='Цвет основного изображения', justify=RIGHT)
color_before_value = Label(main_frame, font=40, justify=LEFT)

color_after_label = Label(main_frame, font=40, text='Цвет обработанного изображения', justify=RIGHT)
color_after_value = Label(main_frame, font=40, justify=LEFT)


def set_always_top():
    main_frame.attributes('-topmost', True if always_top_var.get() else False)
    main_frame.update()


always_top_var = IntVar()
always_top_checkbox = tkinter.Checkbutton(main_frame,
                                          variable=always_top_var,
                                          onvalue=1,
                                          offvalue=0,
                                          command=set_always_top,
                                          text='Поверх окон')


def help_open():
    if help_var.get():
        help_field.grid(row=100, column=1, columnspan=6, sticky=N)
    else:
        help_field.grid_remove()


help_field = Label(main_frame, text='\n'
                                    '=== Информация о приложении:\n'
                                    '\n'
                                    'Версия приложения: 1.1.\n'
                                    '\n'
                                    'Данное приоржение представляет собой пользовательский интерфейс\n'
                                    'для фильтрации изображений со следующим набором функций:\n'
                                    '\n'
                                    '- Выбор между среднегеометрическим и среднегармоническим фильтрами;\n'
                                    '- Выбор размера маски фильтра (размера апертуры) - 3, 5, 7...;\n'
                                    '- Выбор между одномерной и двумерной фильтрациями;\n'
                                    '- Выбор типа границ при наложении фильтра;\n'
                                    '- Выбор цвета константы цвета границ при константном фильтре;\n'
                                    '- Выбор типа преобразования изображения (монохром, rgb, yuv);\n'
                                    '- Загрузка исходного изображения из файловой системы;\n'
                                    '- Предпросмотр исходного изображения;\n'
                                    '- Применение фильтра;\n'
                                    '- Сохранение фильтрованного изображения;\n'
                                    '- Отображение цвета пикселя исходного и обработанного изображения;\n'
                                    '- Визуализация гистрограммы исходного и обработанного изображения;\n'
                                    '- "поверх окон".\n'
                                    '\n'
                                    '===\n',
                   font=24, justify=LEFT)
help_var = IntVar()
help_checkbox = tkinter.Checkbutton(main_frame,
                                    variable=help_var,
                                    onvalue=1,
                                    offvalue=0,
                                    command=help_open,
                                    text='Помощь')


border_type_dict = {'Константа          | iiiiii [ abcdefgh ] iiiiiii |': cv2.BORDER_CONSTANT,
                    'Повторение         | aaaaaa [ abcdefgh ] hhhhhhh |': cv2.BORDER_REPLICATE,
                    'Отражение          | fedcba [ abcdefgh ] hgfedcb |': cv2.BORDER_REFLECT,
                    'Свертка            | cdefgh [ abcdefgh ] abcdefg |': cv2.BORDER_WRAP,
                    'Отражение 101      | gfedcb [ abcdefgh ] gfedcba |': cv2.BORDER_REFLECT_101,
                    'Транспонирование   | uvwxyz [ absdefgh ] ijklmop |': cv2.BORDER_TRANSPARENT, }

original_histogram_button = Button(main_frame, text='гистограмма до     ', command=original_histogram_open, font=40)
filtered_histogram_button = Button(main_frame, text='гистограмма после', command=filtered_histogram_open, font=40)

selected_filter = StringVar()
selected_filter.set('среднегармонический')
filter_select_dropdown = OptionMenu(main_frame, selected_filter, 'среднегармонический', 'среднегеометрический')

dimensional = StringVar()
dimensional.set('двумерный')
dimensional_select_dropdown = OptionMenu(main_frame, dimensional, 'двумерный', 'одномерный горизонтальный',
                                         'одномерный вертикальный')


def open_color_chooser():
    RGB, HEX = askcolor(title="Выбор цвета для константного заполнения")
    if RGB:
        border_constant_color_button['bg'] = HEX

        # Рпределяем яркость цвета фона для изменения цвета текста на черный или белый.
        if rgb_2_gray(*RGB) < 128:
            border_constant_color_button['fg'] = '#ffffff'
        else:
            border_constant_color_button['fg'] = '#000000'


def color_chooser_button(value):
    if border_type_dict[value] == cv2.BORDER_CONSTANT:
        border_constant_color_button['state'] = 'active'
    else:
        border_constant_color_button['state'] = 'disabled'


border_type_var = StringVar()
border_type_var.set(list(border_type_dict.keys())[0])
border_type_dropdown = OptionMenu(main_frame, border_type_var, *border_type_dict.keys(), command=color_chooser_button)
border_constant_color_button = Button(main_frame, command=open_color_chooser, text='Выбрать цвет границы')
border_constant_color_button['bg'] = '#ffffff'


total_time = Label(main_frame)
filter_time = Label(main_frame)


color_label = Label(main_frame, text='Цветная обработка')

filter_select_label = Label(main_frame, text='Фильтр')
dimensional_select_label = Label(main_frame, text='Размерность')
aperture_label = Label(main_frame, text='Апертура')
border_type_label = Label(main_frame, text='Тип границ')

gray_image_dict = {'преобразовать в монохромное': 'gray',
                   'преобразовать каждый канал цвета отдельно': 'rgb',
                   'преобразовать яркостную составляющу (y из yuv)': 'yuv'}

gray_image = StringVar()
gray_image.set(list(gray_image_dict.keys())[0])
gray_image_dropdown = OptionMenu(main_frame, gray_image, *gray_image_dict.keys())
gray_image_label = Label(main_frame, text='Тип обработки')

r_var = BooleanVar()
r_var.set(0)
r1 = Radiobutton(text='First', variable=r_var, value=0)
r2 = Radiobutton(text='Second', variable=r_var, value=1)


original_image_path_field.grid(     row=2, column=2, columnspan=4, sticky=W)
find_image_button.grid(             row=2, column=1, columnspan=1, sticky=W)
show_image_button.grid(             row=2, column=7, columnspan=1, sticky=E)

aperture_label.grid(                row=4, column=1, columnspan=1)
dimensional_select_label.grid(      row=6, column=1, columnspan=1)
filter_select_label.grid(           row=8, column=1, columnspan=1)
border_type_label.grid(             row=10, column=1, columnspan=1)

aperture_scale.grid(                row=4, column=2, columnspan=2, sticky=W)
dimensional_select_dropdown.grid(   row=6, column=2, columnspan=2, sticky=W)
filter_select_dropdown.grid(        row=8, column=2, columnspan=2, sticky=W)
border_type_dropdown.grid(          row=10, column=2, columnspan=3, sticky=W)
border_constant_color_button.grid(  row=11, column=2, columnspan=1, sticky=W)

gray_image_label.grid(              row=9, column=1, columnspan=1, sticky=W)
gray_image_dropdown.grid(           row=9, column=2, columnspan=2, sticky=W)

open_filtered_image_button.grid(    row=4, column=7, columnspan=1, sticky=E)
save_filtered_image_button.grid(    row=6, column=7, columnspan=1, sticky=E)
original_histogram_button.grid(     row=12, column=5, columnspan=3, sticky=E)
filtered_histogram_button.grid(     row=14, column=5, columnspan=3, sticky=E)

color_before_label.grid(            row=12, column=2, columnspan=2, sticky=E)
color_before_value.grid(            row=12, column=4, sticky=W)
color_after_label.grid(             row=14, column=2, columnspan=2, sticky=E)
color_after_value.grid(             row=14, column=4, sticky=W)

always_top_checkbox.grid(           row=8, column=7, columnspan=1, sticky=E)
help_checkbox.grid(                 row=9, column=7)

total_time.grid(                    row=16, column=1, columnspan=1, sticky=W)
filter_time.grid(                   row=16, column=2, columnspan=1, sticky=W)

main_frame.mainloop()
