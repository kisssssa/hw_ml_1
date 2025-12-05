import plotly.express as px
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ARTIFACTS_PATH = Path("artifacts.pkl")

with ARTIFACTS_PATH.open("rb") as f:
    art = pickle.load(f)

model        = art["model"]          # LinearRegression по log(price)
scaler       = art["scaler"]         # StandardScaler
FEATURES     = art["feature_names"]  # колонки после OHE
CAT_COLS     = art["cat_cols"]       # ['name','fuel','seller_type','transmission','owner','seats']
CURRENT_YEAR = art["current_year"]


# 1. ПРЕПРОЦЕССИНГ ПРИЗНАКОВ

def make_features(df_raw: pd.DataFrame) -> pd.DataFrame:

    df = df_raw.copy()

    df["age"] = CURRENT_YEAR - df["year"]
    df["km_per_year"] = df["km_driven"] / df["age"].clip(lower=1)
    df["log_km_driven"] = np.log1p(df["km_driven"])
    df["power_per_litre"] = df["max_power"] / df["engine"]
    df["torque_per_litre"] = df["torque"] / df["engine"]

    return df


def prepare_X(df_raw: pd.DataFrame):
    """
    full pipeline: feature engineering -> OHE -> reindex -> scale
    """
    df = make_features(df_raw)

    # в инференсе таргет не нужен
    df = df.drop(columns=["selling_price", "log_price", "name_clean"], errors="ignore")

    # One-Hot по категориальным
    df_ohe = pd.get_dummies(df, columns=CAT_COLS, dtype=int)

    # выравниваем под набор признаков, с которым обучался model
    df_ohe = df_ohe.reindex(columns=FEATURES, fill_value=0)

    # масштабирование
    X_scaled = scaler.transform(df_ohe)

    X_scaled_df = pd.DataFrame(X_scaled, columns=FEATURES, index=df_ohe.index)

    return X_scaled_df, df_ohe


def predict_prices(df_raw: pd.DataFrame) -> pd.Series:
    """
    Прогноз цены в рублях:
      X -> model (log-price) -> expm1
    """
    X_scaled, _ = prepare_X(df_raw)
    log_pred = model.predict(X_scaled)
    price_pred = np.expm1(log_pred)
    return pd.Series(price_pred, index=df_raw.index, name="pred_price")

#-----------------------------------


# ============================
# 2. EDA-ГРАФИКИ
# ============================

def eda_plots(df_raw: pd.DataFrame):
    """
    EDA с интерактивными графиками
    """
    df = make_features(df_raw.copy())

    has_target = "selling_price" in df.columns
    if has_target:
        df["log_price"] = np.log1p(df["selling_price"])

    st.markdown("### Базовое описание данных")

    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if num_cols:
        st.write(df[num_cols].describe().T)
    else:
        st.info("В данных нет числовых признаков.")

    # 1. Гистограмма любого числового признака
    st.markdown("### Распределение числового признака")

    if num_cols:
        # если есть log_price - по умолчанию его, иначе первый числовой
        default_hist = "log_price" if has_target and "log_price" in num_cols else num_cols[0]

        hist_col = st.selectbox(
            "Числовой признак для гистограммы:",
            num_cols,
            index=num_cols.index(default_hist),
        )

        fig_hist = px.histogram(
            df,
            x=hist_col,
            nbins=50,
            marginal="box",
            title=f"Распределение {hist_col}",
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        st.info("Нет числовых столбцов для гистограммы.")

    # 2. Scatter-графики
    st.markdown("### Взаимосвязи признаков")

    numeric_candidates = [
        c for c in [
            "year", "age",
            "km_driven", "km_per_year",
            "mileage",
            "engine", "max_power", "torque",
            "log_km_driven",
            "power_per_litre", "torque_per_litre",
            "seats",
        ]
        if c in df.columns
    ]

    cat_candidates = [c for c in ["fuel", "seller_type", "transmission", "owner", "name"] if c in df.columns]

    # режим "Y = log_price", только если есть таргет
    if has_target and "log_price" in df.columns and numeric_candidates:
        st.subheader("Связь признака с log(ценой)")

        x_col = st.selectbox(
            "Признак по оси X:",
            numeric_candidates,
            index=numeric_candidates.index("year") if "year" in numeric_candidates else 0,
            key="scatter_x_with_target",
        )

        color_col = st.selectbox(
            "Окрасить точки по категориальному признаку:",
            cat_candidates if cat_candidates else ["<нет>"],
            index=0,
            key="scatter_color_with_target",
        )
        if color_col == "<нет>":
            color_col = None

        fig_scatter = px.scatter(
            df,
            x=x_col,
            y="log_price",
            color=color_col,
            hover_data=[c for c in ["name", "year", "km_driven", "selling_price"] if c in df.columns],
            opacity=0.5,
            trendline="ols",
            title=f"{x_col} vs log(selling_price)",
        )
        st.plotly_chart(fig_scatter, use_container_width=True)

    # scatter между признаками (работает и с таргетом, и без)
    if len(numeric_candidates) >= 2:
        st.subheader("Связь признаков между собой")

        col1, col2 = st.columns(2)
        with col1:
            x_col2 = st.selectbox(
                "Признак по оси X:",
                numeric_candidates,
                index=0,
                key="scatter_x_no_target",
            )
        with col2:
            y_col2 = st.selectbox(
                "Признак по оси Y:",
                numeric_candidates,
                index=1 if len(numeric_candidates) > 1 else 0,
                key="scatter_y_no_target",
            )

        color_col2 = st.selectbox(
            "Окрасить точки по категориальному признаку:",
            cat_candidates if cat_candidates else ["<нет>"],
            index=0,
            key="scatter_color_no_target",
        )
        if color_col2 == "<нет>":
            color_col2 = None

        fig_scatter2 = px.scatter(
            df,
            x=x_col2,
            y=y_col2,
            color=color_col2,
            hover_data=[c for c in ["name", "year", "km_driven"] if c in df.columns],
            opacity=0.5,
            title=f"{x_col2} vs {y_col2}",
        )
        st.plotly_chart(fig_scatter2, use_container_width=True)
    else:
        st.info("Недостаточно числовых признаков для scatter-графика.")

    # 3. Boxplot по категориям (только если есть цена)
    if has_target:
        st.markdown("### Boxplot log(цены) по категориям")

        cat_candidates_box = [c for c in ["fuel", "seller_type", "transmission", "owner", "name"] if c in df.columns]

        if cat_candidates_box:
            cat_col = st.selectbox(
                "Категориальный признак для boxplot:",
                cat_candidates_box,
                index=0,
            )

            df_box = df.copy()
            if cat_col == "name":
                top_brands = df_box["name"].value_counts().head(10).index
                df_box["name_box"] = df_box["name"].where(df_box["name"].isin(top_brands), other="Other")
                cat_plot = "name_box"
            else:
                cat_plot = cat_col

            fig_box = px.box(
                df_box,
                x=cat_plot,
                y="log_price",
                points="all",
                title=f"log(selling_price) по {cat_col}",
            )
            fig_box.update_layout(xaxis_tickangle=-30)
            st.plotly_chart(fig_box, use_container_width=True)
        else:
            st.info("Нет категориальных признаков для boxplot.")

    # 4. Корреляции
    if num_cols:
        if has_target and "log_price" in df.columns:
            st.markdown("### Корреляции числовых признаков")

            # 4.1. Корреляции с log(ценой)
            cors = df[num_cols].corr()["log_price"].sort_values(ascending=False)
            cors_to_show = cors.drop(index=["log_price"])

            fig_corr = px.bar(
                cors_to_show,
                title="Корреляции признаков с log(selling_price)",
                labels={"value": "corr", "index": "feature"},
            )
            st.plotly_chart(fig_corr, use_container_width=True)

            # 4.2. Полная корреляционная матрица
            corr_mat = df[num_cols].corr()
            fig_corr_mat = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                origin="lower",
                title="Корреляционная матрица числовых признаков",
            )
            st.plotly_chart(fig_corr_mat, use_container_width=True)
        else:
            st.markdown("### Корреляция между числовыми признаками")

            corr_mat = df[num_cols].corr()
            fig_corr = px.imshow(
                corr_mat,
                text_auto=".2f",
                aspect="auto",
                color_continuous_scale="RdBu_r",
                origin="lower",
                title="Корреляционная матрица числовых признаков",
            )
            st.plotly_chart(fig_corr, use_container_width=True)




# 3. ВИЗУАЛИЗАЦИЯ КОЭФФИЦИЕНТОВ

def show_coefficients():
    st.header("Коэффициенты модели (лог-цена)")

    # все коэффициенты
    coefs = pd.Series(model.coef_, index=FEATURES, name="coef")

    # Разделим признаки на числовые и категориальные по префиксам OHE
    cat_prefixes = [f"{c}_" for c in CAT_COLS]
    cat_mask = coefs.index.to_series().apply(
        lambda c: any(c.startswith(p) for p in cat_prefixes)
    )

    cat_features = coefs[cat_mask]
    num_features = coefs[~cat_mask]

    #Панель управления
    st.subheader("Фильтры")

    group = st.radio(
        "Какие коэффициенты показать?",
        ["Все", "Только числовые", "Только категориальные"],
        index=0,
        horizontal=True,
    )

    max_n = st.slider(
        "Сколько признаков показать (по модулю)?",
        min_value=5,
        max_value=min(50, len(coefs)),
        value=min(15, len(coefs)),
        step=1,
    )

    show_abs = st.checkbox(
        "Сортировать по модулю (|coef|), но показывать со знаком",
        value=True,
    )

    #Выбор подмножества
    if group == "Все":
        selected = coefs.copy()
    elif group == "Только числовые":
        selected = num_features.copy()
    else:
        selected = cat_features.copy()

    if selected.empty:
        st.warning("Для выбранной группы признаков нет коэффициентов.")
        return

    # сортировка
    if show_abs:
        selected = selected.reindex(selected.abs().sort_values(ascending=False).head(max_n).index)
    else:
        selected = selected.sort_values(ascending=False).head(max_n)

    #Интерактивный бар-чарт
    st.subheader("Бар-чарт коэффициентов")

    df_plot = selected.rename_axis("feature").reset_index()

    fig = px.bar(
        df_plot.sort_values("coef"),
        x="coef",
        y="feature",
        orientation="h",
        hover_data={"coef": ":.4f", "feature": True},
        title="Коэффициенты модели по выбранным признакам",
    )
    fig.update_layout(
        xaxis_title="коэффициент (β в log(price))",
        yaxis_title="признак",
        height=600,
    )
    st.plotly_chart(fig, use_container_width=True)

    # --- Табличка для копания ---
    st.subheader("Таблица коэффициентов")

    df_table = df_plot.copy()
    df_table["abs_coef"] = df_table["coef"].abs()
    df_table = df_table.sort_values("abs_coef", ascending=False)

    st.dataframe(
        df_table[["feature", "coef", "abs_coef"]],
        use_container_width=True,
    )

    st.markdown(
        """
**Интерпретация**:

- Модель обучена на `log(selling_price)`, поэтому коэффициент `β`
  показывает *приблизительное логарифмическое влияние* признака.
- Положительный `β` → рост признака связан с ростом log-цены (и, грубо говоря, с ростом цены в %).
- Отрицательный `β` → рост признака связан с падением log-цены.
- Для дамми-признаков (`fuel_Diesel`, `name_Maruti` и т.п.) коэффициент —
  это эффект **относительно базовой категории** (той, для которой нет отдельной OHE-колонки).
        """
    )


# 4. STREAMLIT UI

st.set_page_config(page_title="Car Price Regression", layout="wide")

st.title("Сервис предсказания цены автомобиля")
st.markdown(
    "Финальная модель: линейная регрессия по **log(selling_price)** "
    "с OHE категориальных признаков и стандартизацией."
)

st.sidebar.header("Навигация")
page = st.sidebar.radio(
    "Страница",
    ["EDA", "Прогнозы", "Коэффициенты"],
)

#загрузка CSV

st.sidebar.header("Данные")

uploaded_file = st.sidebar.file_uploader(
    "Загрузите CSV с данными (сырые признаки + selling_price)",
    type=["csv"],
)

df_data: pd.DataFrame
if uploaded_file is not None:
    df_data = pd.read_csv(uploaded_file)
else:
    st.sidebar.info("Файл не загружен. Можно положить рядом cars_train_for_app.csv.")
    try:
        df_data = pd.read_csv("cars_train_for_app.csv")
    except Exception:
        df_data = pd.DataFrame()
        st.sidebar.error("Нет загруженного файла и не найден cars_train_for_app.csv.")


# 5. ЛОГИКА СТРАНИЦ

if page == "EDA":
    st.header("Exploratory Data Analysis")
    if df_data.empty:
        st.warning("Нет данных для анализа. Загрузите CSV в сайдбаре.")
    else:
        st.write("Размер датасета:", df_data.shape)
        st.dataframe(df_data.head())
        eda_plots(df_data)

elif page == "Прогнозы":
    st.header("Прогнозы по модели")

    if df_data.empty:
        st.warning("Нет данных. Загрузите CSV в сайдбаре.")
    else:
        tab_csv, tab_manual = st.tabs(["Прогноз по CSV", "Ручной ввод"])

        #Прогноз по CSV
        with tab_csv:
            st.subheader("Применить модель к загруженному датасету")

            preds = predict_prices(df_data)
            df_with_pred = df_data.copy()
            df_with_pred["pred_price"] = preds

            if "selling_price" in df_with_pred.columns:
                df_with_pred["abs_pct_error"] = (
                        (df_with_pred["pred_price"] - df_with_pred["selling_price"]).abs()
                        / df_with_pred["selling_price"]
                )

            st.dataframe(df_with_pred.head())

        #Ручной ввод
        with tab_manual:
            st.subheader("Ручной ввод одного автомобиля")

            if df_data.empty:
                # дефолтные значения, если нет примера
                name_list   = []
                fuel_list   = []
                seller_list = []
                trans_list  = []
                owner_list  = []
                seats_list  = []
            else:
                name_list   = sorted(df_data["name"].astype(str).unique())
                fuel_list   = sorted(df_data["fuel"].astype(str).unique())
                seller_list = sorted(df_data["seller_type"].astype(str).unique())
                trans_list  = sorted(df_data["transmission"].astype(str).unique())
                owner_list  = sorted(df_data["owner"].astype(str).unique())
                seats_list  = sorted(df_data["seats"].dropna().unique())

            col1, col2, col3 = st.columns(3)

            with col1:
                year = st.number_input("Год выпуска", 1980, CURRENT_YEAR, 2015)
                km_driven = st.number_input("Пробег, км", 0, 2_000_000, 50_000)
                mileage = st.number_input("Расход (kmpl)", 0.0, 50.0, 18.0)

            with col2:
                max_power = st.number_input("Мощность, bhp", 30.0, 1000.0, 100.0)
                engine = st.number_input("Объём, cc", 600, 6000, 1500)
                torque = st.number_input("Крутящий момент, Nm", 50.0, 800.0, 150.0)

            with col3:
                if name_list:
                    name = st.selectbox("Марка/модель (name)", name_list)
                else:
                    name = st.text_input("Марка/модель (name)", "Maruti")

                if fuel_list:
                    fuel = st.selectbox("Топливо (fuel)", fuel_list)
                else:
                    fuel = st.text_input("Топливо (fuel)", "Petrol")

                if seller_list:
                    seller_type = st.selectbox("Тип продавца (seller_type)", seller_list)
                else:
                    seller_type = st.text_input("seller_type", "Dealer")

                if trans_list:
                    transmission = st.selectbox("КПП (transmission)", trans_list)
                else:
                    transmission = st.text_input("transmission", "Manual")

                if owner_list:
                    owner = st.selectbox("Владельцы (owner)", owner_list)
                else:
                    owner = st.text_input("owner", "First Owner")

                if seats_list:
                    seats = st.selectbox("Мест (seats)", seats_list)
                else:
                    seats = st.number_input("Мест (seats)", 2, 10, 5)

            if st.button("Посчитать цену"):
                row = {
                    "name": name,
                    "year": year,
                    "km_driven": km_driven,
                    "mileage": mileage,
                    "engine": engine,
                    "max_power": max_power,
                    "torque": torque,
                    "seats": seats,
                    "fuel": fuel,
                    "seller_type": seller_type,
                    "transmission": transmission,
                    "owner": owner,
                }
                df_one = pd.DataFrame([row])
                price_pred = predict_prices(df_one).iloc[0]
                st.success(f"Прогнозируемая цена: **{price_pred:,.0f}**")

elif page == "Коэффициенты":
    show_coefficients()
