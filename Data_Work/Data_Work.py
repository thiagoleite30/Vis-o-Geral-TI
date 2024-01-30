from datetime import date as dt
from dateutil.relativedelta import relativedelta


class Data_Work:

    def __init__(self) -> None:
        pass

    def get_data_range_open(self, start_date, end_date, ano_comparacao):
        # Primeiro pega a diferença entre os anos do intervalo
        dif_years = dt.fromisoformat(end_date).year - ano_comparacao

        # Monta a string do periodo do ano atual
        data_atual_end = (dt.fromisoformat(
            end_date).strftime('%Y-%m-%d')) + ' 23:59:59-03:00'
        data_atual_start = dt.fromisoformat(
            start_date).strftime('%Y-%m-%d') + ' 00:00:00-03:00'
        # print(f'Data atual end: {data_atual_end}')
        # print(f'Data atual start: {data_atual_start}')

        # Monta a string do periodo do ano anterior
        data_anterior_end = (dt.fromisoformat(
            end_date) - relativedelta(years=dif_years)).strftime('%Y-%m-%d') + ' 23:59:59-03:00'
        data_anterior_start = (dt.fromisoformat(
            start_date) - relativedelta(years=dif_years)).strftime('%Y-%m-%d') + ' 00:00:00-03:00'

        return f'(DATA_ABERTURA >= "{data_atual_start}" and DATA_ABERTURA <= "{data_atual_end}") or (DATA_ABERTURA >= "{data_anterior_start}" and DATA_ABERTURA <= "{data_anterior_end}")'

    def get_data_range_closed(self, start_date, end_date, ano_comparacao):
        # Primeiro pega a diferença entre os anos do intervalo
        dif_years = dt.fromisoformat(end_date).year - ano_comparacao

        # Monta a string do periodo do ano atual
        data_atual_end = (dt.fromisoformat(
            end_date).strftime('%Y-%m-%d')) + ' 23:59:59-03:00'
        data_atual_start = dt.fromisoformat(
            start_date).strftime('%Y-%m-%d') + ' 00:00:00-03:00'
        # print(f'Data atual end: {data_atual_end}')
        # print(f'Data atual start: {data_atual_start}')

        # Monta a string do periodo do ano anterior
        data_anterior_end = (dt.fromisoformat(
            end_date) - relativedelta(years=dif_years)).strftime('%Y-%m-%d') + ' 23:59:59-03:00'
        data_anterior_start = (dt.fromisoformat(
            start_date) - relativedelta(years=dif_years)).strftime('%Y-%m-%d') + ' 00:00:00-03:00'
        return f'(DATA_FECHAMENTO >= "{data_atual_start}" and DATA_FECHAMENTO <= "{data_atual_end}") or (DATA_FECHAMENTO >= "{data_anterior_start}" and DATA_FECHAMENTO <= "{data_anterior_end}")'

    def get_data_open(self, start_date, end_date):
        # Primeiro pega a diferença entre os anos do intervalo

        # Monta a string do periodo do ano atual
        date_end = (dt.fromisoformat(
            end_date).strftime('%Y-%m-%d')) + ' 23:59:59-03:00'
        date_start = dt.fromisoformat(
            start_date).strftime('%Y-%m-%d') + ' 00:00:00-03:00'
        # print(f'Data atual end: {data_atual_end}')
        # print(f'Data atual start: {data_atual_start}')

        return f'(DATA_ABERTURA >= "{date_start}" and DATA_ABERTURA <= "{date_end}")'
