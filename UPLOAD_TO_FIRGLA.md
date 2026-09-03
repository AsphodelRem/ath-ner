# Что загрузить в ветку `firgla`

Готовый компактный пакет лежит рядом в архиве `firgla_upload.zip` и папке
`upload_to_firgla/`.

1. Распаковать `firgla_upload.zip`.
2. На GitHub переключиться на уже созданную ветку `firgla`.
3. Нажать **Add file → Upload files**.
4. Перетащить **содержимое** папки `upload_to_firgla`, чтобы `README.md`,
   `baseline/`, `final/` и остальные пути оказались в корне репозитория.
5. Commit message: `Add BILOU Viterbi two-seed NER solution`.

Не загружать `.venv/`, `artifacts/`, `data/`, папку `upload_to_firgla` целиком
как дополнительный уровень или сам zip-файл. Большие веса находятся в
`artifacts/` локально и воспроизводятся по `final/README.md`.
