install postgresql or monodb

sudo apt install postgresql postgresql-contrib

sudo systemctl start postgresql
sudo systemctl enable postgresql

psql --version

sudo -u postgres psql

CREATE DATABASE fire_robot;
CREATE USER admin WITH PASSWORD 'yourpassword';
GRANT ALL PRIVILEGES ON DATABASE fire_robot TO admin;
\q