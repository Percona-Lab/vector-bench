-- Runs via --init-file on every VillageSQL start, and once during
-- initialisation. Must be idempotent.
--
-- Identical in effect to docker/percona/init.sql and docker/myvector/init.sql:
-- the default (caching_sha2_password). MySQL 9.7 has removed
-- mysql_native_password from the server, and villagesql84's MySQL 8.4 base
-- ships it disabled, so both VillageSQL engines authenticate the same way and
-- run this file unchanged. The extension is NOT installed
-- here -- INSTALL EXTENSION is not idempotent and runs once, from the
-- entrypoint's bootstrap.
CREATE USER IF NOT EXISTS 'bench'@'%' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'%' WITH GRANT OPTION;
CREATE USER IF NOT EXISTS 'bench'@'localhost' IDENTIFIED BY 'bench';
GRANT ALL PRIVILEGES ON *.* TO 'bench'@'localhost' WITH GRANT OPTION;
FLUSH PRIVILEGES;
