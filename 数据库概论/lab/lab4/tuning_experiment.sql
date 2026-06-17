/*
数据库性能调优实验脚本
适用库：db_proj4
运行方式：
  mysql -u root -p --default-character-set=utf8mb4 db_proj4 < tuning_experiment.sql

说明：
  1. 为了保证“优化前/优化后”对比清晰，脚本会删除并重建本实验使用的索引。
  2. 事务调优、模式调优部分使用实验副本表，不破坏原始业务表数据。
*/

SET NAMES utf8mb4;
USE db_proj4;

/* =========================================================
0. 环境检查与辅助过程
========================================================= */
DROP PROCEDURE IF EXISTS drop_index;
DROP PROCEDURE IF EXISTS ensure_index;

DELIMITER //
CREATE PROCEDURE drop_index(
    IN db_name VARCHAR(64),
    IN target_table VARCHAR(64),
    IN target_index VARCHAR(64)
)
BEGIN
    IF EXISTS (
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = db_name
          AND table_name = target_table
          AND index_name = target_index
    ) THEN
        SET @sql_text = CONCAT('DROP INDEX `', target_index, '` ON `', target_table, '`');
        PREPARE stmt FROM @sql_text;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END //

CREATE PROCEDURE ensure_index(
    IN db_name VARCHAR(64),
    IN target_table VARCHAR(64),
    IN target_index VARCHAR(64),
    IN create_sql TEXT
)
BEGIN
    IF NOT EXISTS (
        SELECT 1
        FROM information_schema.statistics
        WHERE table_schema = db_name
          AND table_name = target_table
          AND index_name = target_index
    ) THEN
        SET @sql_text = create_sql;
        PREPARE stmt FROM @sql_text;
        EXECUTE stmt;
        DEALLOCATE PREPARE stmt;
    END IF;
END //
DELIMITER ;

SELECT '0. 环境与数据量检查' AS step;
SELECT VERSION() AS mysql_version, DATABASE() AS database_name;
SELECT 'user_info' AS table_name, COUNT(*) AS row_count FROM user_info
UNION ALL
SELECT 'goods', COUNT(*) FROM goods
UNION ALL
SELECT 'order_main', COUNT(*) FROM order_main
UNION ALL
SELECT 'order_item', COUNT(*) FROM order_item;

/* =========================================================
题目1：单字段索引设计
模块：索引调优
========================================================= */
SELECT '题目1：单字段索引设计' AS step;

CALL drop_index(DATABASE(), 'user_info', 'idx_user_info_phone');
CALL drop_index(DATABASE(), 'user_info', 'idx_user_info_email');
ANALYZE TABLE user_info;

SET @phone := (SELECT phone FROM user_info WHERE user_id = 100000);
SET @email := (SELECT email FROM user_info WHERE user_id = 100000);
SELECT @phone AS sample_phone, @email AS sample_email;

SELECT '题目1-优化前：phone 精准查询，无索引' AS case_name;
EXPLAIN SELECT user_id, nickname, phone, email
FROM user_info
WHERE phone = @phone;
EXPLAIN ANALYZE SELECT user_id, nickname, phone, email
FROM user_info
WHERE phone = @phone;

SELECT '题目1-优化前：email 精准查询，无索引' AS case_name;
EXPLAIN SELECT user_id, nickname, phone, email
FROM user_info
WHERE email = @email;
EXPLAIN ANALYZE SELECT user_id, nickname, phone, email
FROM user_info
WHERE email = @email;

CALL ensure_index(
    DATABASE(),
    'user_info',
    'idx_user_info_phone',
    'CREATE INDEX idx_user_info_phone ON user_info(phone)'
);
CALL ensure_index(
    DATABASE(),
    'user_info',
    'idx_user_info_email',
    'CREATE INDEX idx_user_info_email ON user_info(email)'
);
ANALYZE TABLE user_info;

SELECT '题目1-优化后：phone 命中单字段索引' AS case_name;
EXPLAIN SELECT user_id, nickname, phone, email
FROM user_info
WHERE phone = @phone;
EXPLAIN ANALYZE SELECT user_id, nickname, phone, email
FROM user_info
WHERE phone = @phone;

SELECT '题目1-优化后：email 命中单字段索引' AS case_name;
EXPLAIN SELECT user_id, nickname, phone, email
FROM user_info
WHERE email = @email;
EXPLAIN ANALYZE SELECT user_id, nickname, phone, email
FROM user_info
WHERE email = @email;

/* =========================================================
题目6：函数操作字段导致索引失效
模块：索引调优
========================================================= */
SELECT '题目6：函数操作字段导致索引失效' AS step;

CALL ensure_index(
    DATABASE(),
    'order_main',
    'idx_order_main_create_time',
    'CREATE INDEX idx_order_main_create_time ON order_main(create_time)'
);
ANALYZE TABLE order_main;

SET @day := '2026-03-15';
SELECT @day AS sample_day;

SELECT '题目6-优化前：DATE(create_time) 包裹索引列' AS case_name;
EXPLAIN SELECT order_id, create_time
FROM order_main
WHERE DATE(create_time) = @day;
EXPLAIN ANALYZE SELECT order_id, create_time
FROM order_main
WHERE DATE(create_time) = @day;

SELECT '题目6-优化后：半开区间范围查询' AS case_name;
EXPLAIN SELECT order_id, create_time
FROM order_main
WHERE create_time >= @day
  AND create_time < DATE_ADD(@day, INTERVAL 1 DAY);
EXPLAIN ANALYZE SELECT order_id, create_time
FROM order_main
WHERE create_time >= @day
  AND create_time < DATE_ADD(@day, INTERVAL 1 DAY);

/* =========================================================
题目9：杜绝 SELECT *
模块：SQL 语句调优
========================================================= */
SELECT '题目9：杜绝 SELECT * + 覆盖索引' AS step;

CALL drop_index(DATABASE(), 'order_main', 'idx_order_main_create_time');
CALL drop_index(DATABASE(), 'order_main', 'idx_om_status_ct_cover');
ANALYZE TABLE order_main;

SET @status := '已完成';
SELECT @status AS sample_status;

SELECT '题目9-优化前：SELECT * 读取整行并触发额外排序' AS case_name;
EXPLAIN SELECT *
FROM order_main
WHERE order_status = @status
ORDER BY create_time DESC
LIMIT 100;
EXPLAIN ANALYZE SELECT *
FROM order_main
WHERE order_status = @status
ORDER BY create_time DESC
LIMIT 100;

CALL ensure_index(
    DATABASE(),
    'order_main',
    'idx_om_status_ct_cover',
    'CREATE INDEX idx_om_status_ct_cover ON order_main(order_status, create_time, order_id, amount)'
);
ANALYZE TABLE order_main;

SELECT '题目9-优化后：按需字段 + 覆盖索引' AS case_name;
EXPLAIN SELECT order_id, amount, create_time
FROM order_main
WHERE order_status = @status
ORDER BY create_time DESC
LIMIT 100;
EXPLAIN ANALYZE SELECT order_id, amount, create_time
FROM order_main
WHERE order_status = @status
ORDER BY create_time DESC
LIMIT 100;

/* =========================================================
题目13：Using filesort 文件排序优化
模块：SQL 语句调优
========================================================= */
SELECT '题目13：Using filesort 文件排序优化' AS step;

CALL drop_index(DATABASE(), 'order_main', 'idx_om_user_ct');
ANALYZE TABLE order_main;

SET @sample_user_id := (
    SELECT user_id
    FROM order_main
    GROUP BY user_id
    ORDER BY COUNT(*) DESC
    LIMIT 1
);
SELECT @sample_user_id AS sample_user_id;

SELECT '题目13-优化前：只有 user_id 单列索引，按 create_time 排序仍需 filesort' AS case_name;
EXPLAIN SELECT order_id, user_id, amount, create_time
FROM order_main
WHERE user_id = @sample_user_id
ORDER BY create_time DESC
LIMIT 20;
EXPLAIN ANALYZE SELECT order_id, user_id, amount, create_time
FROM order_main
WHERE user_id = @sample_user_id
ORDER BY create_time DESC
LIMIT 20;

CALL ensure_index(
    DATABASE(),
    'order_main',
    'idx_om_user_ct',
    'CREATE INDEX idx_om_user_ct ON order_main(user_id, create_time, order_id, amount)'
);
ANALYZE TABLE order_main;

SELECT '题目13-优化后：联合索引同时满足过滤、排序和返回字段' AS case_name;
EXPLAIN SELECT order_id, user_id, amount, create_time
FROM order_main
WHERE user_id = @sample_user_id
ORDER BY create_time DESC
LIMIT 20;
EXPLAIN ANALYZE SELECT order_id, user_id, amount, create_time
FROM order_main
WHERE user_id = @sample_user_id
ORDER BY create_time DESC
LIMIT 20;

/* =========================================================
题目17：批量 DML 语句优化
模块：事务调优
========================================================= */
SELECT '题目17：批量 DML 语句优化' AS step;

DROP TABLE IF EXISTS order_insert_demo;
CREATE TABLE order_insert_demo (
    order_id BIGINT PRIMARY KEY AUTO_INCREMENT,
    user_id BIGINT NOT NULL,
    amount DECIMAL(10,2) NOT NULL,
    order_status VARCHAR(10) NOT NULL,
    pay_status VARCHAR(10) NOT NULL,
    create_time DATETIME NOT NULL,
    pay_time VARCHAR(20),
    address VARCHAR(200)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='事务调优插入实验表';

DROP TEMPORARY TABLE IF EXISTS numbers_1_to_10000;
CREATE TEMPORARY TABLE numbers_1_to_10000 (n INT PRIMARY KEY);
INSERT INTO numbers_1_to_10000(n)
SELECT ones.n + tens.n * 10 + hundreds.n * 100 + thousands.n * 1000 + 1 AS n
FROM (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) ones
CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) tens
CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) hundreds
CROSS JOIN (SELECT 0 n UNION ALL SELECT 1 UNION ALL SELECT 2 UNION ALL SELECT 3 UNION ALL SELECT 4 UNION ALL SELECT 5 UNION ALL SELECT 6 UNION ALL SELECT 7 UNION ALL SELECT 8 UNION ALL SELECT 9) thousands;

DROP PROCEDURE IF EXISTS insert_demo_order;
DELIMITER //
CREATE PROCEDURE insert_demo_order(IN row_count INT)
BEGIN
    DECLARE current_row INT DEFAULT 1;
    WHILE current_row <= row_count DO
        INSERT INTO order_insert_demo(user_id, amount, order_status, pay_status, create_time, pay_time, address)
        VALUES (
            FLOOR(1 + RAND() * 500000),
            ROUND(100 + RAND() * 9900, 2),
            ELT(FLOOR(1 + RAND() * 4), '待付款', '待发货', '已发货', '已完成'),
            ELT(FLOOR(1 + RAND() * 2), '未支付', '已支付'),
            DATE_ADD('2026-01-01', INTERVAL FLOOR(RAND() * 120) DAY),
            NULL,
            CONCAT('测试地址', current_row)
        );
        COMMIT;
        SET current_row = current_row + 1;
    END WHILE;
END //
DELIMITER ;

SET @original_autocommit := @@autocommit;
SET SESSION autocommit = 0;
SET @start_time := NOW(6);
CALL insert_demo_order(5000);
SET @row_by_row_ms := TIMESTAMPDIFF(MICROSECOND, @start_time, NOW(6)) / 1000;
SET SESSION autocommit = @original_autocommit;
SELECT COUNT(*) AS row_count_after_row_by_row, @row_by_row_ms AS row_by_row_ms
FROM order_insert_demo;

TRUNCATE TABLE order_insert_demo;

SET @original_autocommit := @@autocommit;
SET SESSION autocommit = 0;
SET @start_time := NOW(6);
INSERT INTO order_insert_demo(user_id, amount, order_status, pay_status, create_time, pay_time, address)
SELECT
    FLOOR(1 + RAND(n) * 500000),
    ROUND(100 + RAND(n + 1) * 9900, 2),
    ELT(FLOOR(1 + RAND(n + 2) * 4), '待付款', '待发货', '已发货', '已完成'),
    ELT(FLOOR(1 + RAND(n + 3) * 2), '未支付', '已支付'),
    DATE_ADD('2026-01-01', INTERVAL FLOOR(RAND(n + 4) * 120) DAY),
    NULL,
    CONCAT('测试地址', n)
FROM numbers_1_to_10000
WHERE n <= 5000;
COMMIT;
SET @batch_ms := TIMESTAMPDIFF(MICROSECOND, @start_time, NOW(6)) / 1000;
SET SESSION autocommit = @original_autocommit;
SELECT COUNT(*) AS row_count_after_batch, @batch_ms AS batch_ms
FROM order_insert_demo;

SELECT
    @row_by_row_ms AS row_by_row_ms,
    @batch_ms AS batch_ms,
    ROUND(@row_by_row_ms / NULLIF(@batch_ms, 0), 2) AS speedup_ratio;

/* =========================================================
题目23：大字段冷热分离设计
模块：模式调优
========================================================= */
SELECT '题目23：大字段冷热分离设计' AS step;

DROP TABLE IF EXISTS goods_hot;
DROP TABLE IF EXISTS goods_detail_ext;

CALL ensure_index(
    DATABASE(),
    'goods',
    'idx_goods_category_ct',
    'CREATE INDEX idx_goods_category_ct ON goods(category_id, create_time)'
);
ANALYZE TABLE goods;

SET @category_id := 3;
SELECT @category_id AS sample_category_id;

SELECT '题目23-优化前：商品列表 SELECT * 携带 TEXT 大字段' AS case_name;
EXPLAIN SELECT *
FROM goods
WHERE category_id = @category_id
ORDER BY create_time DESC
LIMIT 100;
EXPLAIN ANALYZE SELECT *
FROM goods
WHERE category_id = @category_id
ORDER BY create_time DESC
LIMIT 100;

CREATE TABLE goods_hot (
    goods_id BIGINT PRIMARY KEY,
    goods_name VARCHAR(100) NOT NULL,
    price DECIMAL(10,2) NOT NULL,
    stock INT NOT NULL DEFAULT 0,
    category_id BIGINT,
    create_time DATETIME NOT NULL,
    update_time VARCHAR(20),
    KEY idx_goods_hot_category_ct(category_id, create_time)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品热数据表';

CREATE TABLE goods_detail_ext (
    goods_id BIGINT PRIMARY KEY,
    goods_detail TEXT,
    KEY idx_goods_detail_goods_id(goods_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品详情冷数据表';

INSERT INTO goods_hot(goods_id, goods_name, price, stock, category_id, create_time, update_time)
SELECT goods_id, goods_name, price, stock, category_id, create_time, update_time
FROM goods;

INSERT INTO goods_detail_ext(goods_id, goods_detail)
SELECT goods_id, goods_detail
FROM goods;

ANALYZE TABLE goods_hot, goods_detail_ext;

SELECT '题目23-优化后：商品列表只访问热数据表' AS case_name;
EXPLAIN SELECT goods_id, goods_name, price, stock, category_id, create_time
FROM goods_hot
WHERE category_id = @category_id
ORDER BY create_time DESC
LIMIT 100;
EXPLAIN ANALYZE SELECT goods_id, goods_name, price, stock, category_id, create_time
FROM goods_hot
WHERE category_id = @category_id
ORDER BY create_time DESC
LIMIT 100;

SELECT '题目23-详情页：需要大字段时按主键访问冷数据表' AS case_name;
SET @sample_goods_id := (SELECT goods_id FROM goods_hot WHERE category_id = @category_id LIMIT 1);
EXPLAIN SELECT h.goods_id, h.goods_name, h.price, d.goods_detail
FROM goods_hot h
JOIN goods_detail_ext d ON d.goods_id = h.goods_id
WHERE h.goods_id = @sample_goods_id;
EXPLAIN ANALYZE SELECT h.goods_id, h.goods_name, h.price, d.goods_detail
FROM goods_hot h
JOIN goods_detail_ext d ON d.goods_id = h.goods_id
WHERE h.goods_id = @sample_goods_id;

/* =========================================================
补充：查看本实验创建的索引和实验表
========================================================= */
SELECT '补充：查看索引' AS step;
SHOW INDEX FROM user_info;
SHOW INDEX FROM order_main;
SHOW INDEX FROM goods;
SHOW INDEX FROM goods_hot;

SELECT '补充：查看实验副本表数据量' AS step;
SELECT 'order_insert_demo' AS table_name, COUNT(*) AS row_count FROM order_insert_demo
UNION ALL
SELECT 'goods_hot', COUNT(*) FROM goods_hot
UNION ALL
SELECT 'goods_detail_ext', COUNT(*) FROM goods_detail_ext;
