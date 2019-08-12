from airflow import DAG
from airflow.operators.bash_operator import BashOperator
from airflow.operators.python_operator import PythonOperator
from datetime import datetime, timedelta
from pipeline_utils import prepare_data, predict, postprocess



default_args = {
    'owner': 'airflow',
    'depends_on_past': False,
    'start_date': datetime(2015, 6, 1),
    'email': ['test@test.ru'],
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 1,
    'retry_delay': timedelta(minutes=5),
}

dag = DAG('img_pipeline_dag', default_args=default_args, schedule_interval=timedelta(days=1))

t1 = PythonOperator(
    task_id='prepare_data',
    python_callable='prepare_data',
    dag=dag)

t2 = PythonOperator(
    task_id='predict',
    python_callable='predict',
    dag=dag)

t3 = PythonOperator(
    task_id='predict',
    python_callable='predict',
    dag=dag)


t1 >> t2 >> t3