from enum import Enum

from nltk.tokenize import sent_tokenize

from base import MaskFn


class MaskEntityType(Enum):
    NAME = 0
    PROFESSION = 1
    LOCATION = 2
    AGE = 3
    DATE = 4
    ID = 5
    CONTACT = 6


class PersonalEntityMaskFn(MaskFn):
    def __init__(self, is_markup=True, markup_model=None):
        """
        :param is_markup: размечены ли документы на сущности, содержащие личную информацию
        :param markup_model: модель для разметки сущностей с личной информацией
        """
        try:
            sent_tokenize('Ensure punkt installed.')
        except:
            raise ValueError('Need to call nltk.download(\'punkt\')')
        self.is_markup = is_markup
        self.markup_model = markup_model

    @classmethod
    def mask_types(cls):
        return list(MaskEntityType)

    @classmethod
    def mask_type_serialize(cls, m_type):
        return m_type.name.lower()

    def mask(self, doc, is_markup=None):
        """
        Маскирует все размеченные сущности с личной информацией в документе.
        :param doc: текст документа в формате строки
        :param is_markup: размечен ли, поданный документ; если разметки нет, она осуществляется с помощью специальной модели
        :return: Список троек
        (тип замаскированного объекта, сдвиг на начало замаскированного объекта, длина замаскированного объекта)
        """

        is_markup = self.is_markup if is_markup is None else is_markup

        masked_spans = []
        return masked_spans
