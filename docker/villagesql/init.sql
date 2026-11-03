-- Runs via --init-file on every VillageSQL start, and once during
-- initialisation. Must be idempotent.
--
-- Identical in effect to docker/percona/init.sql and docker/myvector/init.sql,
-- the two other MySQL 9.7.2-based servers here: MySQL 9.7 has removed
-- mysql_native_password from the server, so the default
-- (caching_sha2_password) is the only choice. The extension is NOT installed
-- here -- INSTALL EXTENSION is not idempotent and runs once, from the
-- entrypoint's bootstrap.
CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
