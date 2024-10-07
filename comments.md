Привет! Оставил тут комментарии по возникшим недочетам.

1. Об [transformers.py](services/ml_service/scripts/transformers.py)  
    Пришлось использовать этот файл из-за того, что хоть pipeline это обьект `Pipeline`, но всё равно требуется подгрузка классов трансформеров, которые я реализовал сам. По хорошему надо было бы создать package из этих трансформеров и потом качать его, но я в виду не хватки времени не стал этого делать. Хотя возможно я чего то не знаю и можно сохранить было pipeline так, чтобы не приходилось добавлять этот файл. Пока что пусть останется как есть...
2. Об [fast_api_handler.py](services/ml_service/scripts/fast_api_handler.py)  
    Я бы сто процентов расскидал исключения на составляющие (как я это начал делать недавно), но в виду не хватки времени опять же не стал сильно заморачиваться. Раз уж и так дедлайн прошел, то я в спокойном режиме исправил + добавил обработку для загрузки моделей.
3. По поводу кучи скриптов  
    Да, возможно переборщил. По итогу решил так упростить:
    - [callbacks.py](services/ml_service/scripts/callbacks.py) в [limiters.py](services/ml_service/scripts/limiters.py)
    - [prometheus_metrics.py](services/ml_service/scripts/prometheus_metrics.py) в [metrics.py](services/ml_service/scripts/metrics.py)
    - [generators.py](services/ml_service/scripts/generators.py) в [utils.py](services/ml_service/scripts/utils.py). Пришлось добавить `generate_random_model_params` в [settings.py](services/ml_service/scripts/settings.py) чтобы избежать циклической зависимости.
4. По поводу `PYTHONPATH` в [test.py](services/ml_service/tests/test.py)  
    В общем я то добавлял `PYTHONPATH` через [.env](services/.env) и просто забыл убрать `sys.path.append(str(Path(__file__).parent.parent))`. Оказалось, что я криво указал `PYTHONPATH`. По итогу я решил лучше сделать через [setup.py](services/ml_service/setup.py) без `PYTHONPATH`
5. По поводу упрощения кода в [test.py](services/ml_service/tests/test.py)  
    На скорости делал, даже особо не задумался об этом...
6. По поводу `name == 'main'` [test.py](services/ml_service/tests/test.py)  
    Тут моя ошибка, исправил

PS: По какой то причине при создании сервиса для `grafana`, user и пароль у меня сохраняются внутри контейнера с кавычками - то есть `"admin"` вместо `admin`. Не смог понять почему так происходит. Соответственно, нужно добавлять их чтобы авторизоваться в `grafana`.