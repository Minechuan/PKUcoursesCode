-- 注意事项一：Order_item表里含有外码，所以一定要保证前面的表的数据插入完整
-- 注意事项二：MySQL 默认连接时间 30 秒，超时则会主动断开连接，可以通过下面的语句修改连接时间。
SET GLOBAL net_read_timeout = 1000;
SET GLOBAL net_write_timeout = 1000;

-- 1. 用户表（user_info）
CREATE TABLE user_info (
    user_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '用户ID',
    nickname VARCHAR(50) NOT NULL COMMENT '用户昵称',
    phone VARCHAR(11) NOT NULL COMMENT '手机号（字符串类型，故意留隐患）',
    email VARCHAR(50) COMMENT '邮箱',
    register_time VARCHAR(20) NOT NULL COMMENT '注册时间（字符串类型，留优化隐患）',
    age INT COMMENT '年龄',
    status VARCHAR(10) DEFAULT '正常' COMMENT '状态（字符串，留优化隐患）',
    address VARCHAR(200) COMMENT '收货地址'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='用户信息表';

-- 2. 商品表（goods）
CREATE TABLE goods (
    goods_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '商品ID',
    goods_name VARCHAR(100) NOT NULL COMMENT '商品名称',
    goods_detail TEXT COMMENT '商品详情（大文本，留分表隐患）',
    price DECIMAL(10,2) NOT NULL COMMENT '商品价格',
    stock INT NOT NULL DEFAULT 0 COMMENT '库存',
    category_id BIGINT COMMENT '分类ID',
    create_time DATETIME NOT NULL COMMENT '创建时间',
    update_time VARCHAR(20) COMMENT '更新时间（字符串，留隐患）'
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='商品表';

-- 3. 订单主表（order_main）
CREATE TABLE order_main (
    order_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '订单ID',
    user_id BIGINT NOT NULL COMMENT '用户ID',
    amount DECIMAL(10,2) NOT NULL COMMENT '订单金额',
    order_status VARCHAR(10) NOT NULL COMMENT '订单状态（字符串，留隐患）',
    pay_status VARCHAR(10) NOT NULL COMMENT '支付状态',
    create_time DATETIME NOT NULL COMMENT '下单时间',
    pay_time VARCHAR(20) COMMENT '支付时间（字符串，留隐患）',
    address VARCHAR(200) COMMENT '收货地址',
    -- 故意不建索引，留调优空间
    KEY idx_user_id (user_id) -- 仅建基础索引，其他索引缺失
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单主表';

-- 4. 订单项表（order_item）
CREATE TABLE order_item (
    item_id BIGINT PRIMARY KEY AUTO_INCREMENT COMMENT '订单项ID',
    order_id BIGINT NOT NULL COMMENT '订单ID',
    goods_id BIGINT NOT NULL COMMENT '商品ID',
    goods_num INT NOT NULL DEFAULT 1 COMMENT '商品数量',
    goods_price DECIMAL(10,2) NOT NULL COMMENT '商品单价',
    -- 故意不建关联索引，留调优空间
    FOREIGN KEY (order_id) REFERENCES order_main(order_id),
    FOREIGN KEY (goods_id) REFERENCES goods(goods_id)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COMMENT='订单项表';

-- 注意：执行前确保已创建上述4张表，执行过程可能需要5-10分钟（视电脑配置）
-- 1. 关闭安全更新模式（避免执行时报错）
SET SQL_SAFE_UPDATES = 0;

-- 2. 生成user_info（50万条）
DELIMITER //
CREATE PROCEDURE generate_user_info()
BEGIN
    DECLARE i INT DEFAULT 1;
    WHILE i <= 500000 DO
        INSERT INTO user_info (nickname, phone, email, register_time, age, status, address)
        VALUES (
            CONCAT('用户', i),
            CONCAT('138', LPAD(FLOOR(RAND()*100000000), 8, '0')), -- 随机手机号（字符串）
            CONCAT('user', i, '@example.com'),
            CONCAT('202', FLOOR(3+RAND()*3), '-', LPAD(FLOOR(1+RAND()*12), 2, '0'), '-', LPAD(FLOOR(1+RAND()*28), 2, '0')), -- 字符串时间
            FLOOR(18+RAND()*50), -- 随机年龄18-68
            ELT(FLOOR(1+RAND()*3), '正常', '冻结', '注销'), -- 状态
            CONCAT('省份', FLOOR(1+RAND()*34), '城市', FLOOR(1+RAND()*100), '街道', FLOOR(1+RAND()*1000))
        );
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;
CALL generate_user_info();
DROP PROCEDURE generate_user_info;

select count(*) from user_info;

-- 3. 生成goods（10万条）
DELIMITER //
CREATE PROCEDURE generate_goods()
BEGIN
    DECLARE i INT DEFAULT 1;
    WHILE i <= 100000 DO
        INSERT INTO goods (goods_name, goods_detail, price, stock, category_id, create_time, update_time)
        VALUES (
            CONCAT(ELT(FLOOR(1+RAND()*5), '手机', '电脑', '耳机', '手表', '平板'), i), -- 商品名称
            CONCAT('商品', i, '详情：这是一款高性能的电子产品，质量可靠，性价比高，适合各类人群使用，支持全国联保，售后无忧。'), -- 大文本
            ROUND(100 + RAND()*9900, 2), -- 价格100-10000
            FLOOR(100 + RAND()*10000), -- 库存100-10100
            FLOOR(1+RAND()*10), -- 分类ID 1-10
            DATE_ADD('2025-01-01', INTERVAL FLOOR(RAND()*365) DAY), -- 2025年随机创建时间
            CONCAT('202', FLOOR(5+RAND()*1), '-', LPAD(FLOOR(1+RAND()*12), 2, '0'), '-', LPAD(FLOOR(1+RAND()*28), 2, '0')) -- 字符串更新时间
        );
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;
CALL generate_goods();
DROP PROCEDURE generate_goods;

-- 4. 生成order_main（200万条）
DELIMITER //
CREATE PROCEDURE generate_order_main()
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE user_id INT;
    WHILE i <= 2000000 DO
        SET user_id = FLOOR(1+RAND()*500000); -- 关联user_info的user_id
        INSERT INTO order_main (user_id, amount, order_status, pay_status, create_time, pay_time, address)
        VALUES (
            user_id,
            ROUND(100 + RAND()*9900, 2), -- 订单金额100-10000
            ELT(FLOOR(1+RAND()*4), '待付款', '待发货', '已发货', '已完成'), -- 订单状态（字符串）
            ELT(FLOOR(1+RAND()*2), '未支付', '已支付'), -- 支付状态
            DATE_ADD('2026-01-01', INTERVAL FLOOR(RAND()*120) DAY), -- 2026年1-4月随机下单时间
            IF(FLOOR(1+RAND()*2)=2, DATE_ADD(DATE_ADD('2026-01-01', INTERVAL FLOOR(RAND()*120) DAY), INTERVAL FLOOR(1+RAND()*60) MINUTE), NULL), -- 支付时间（随机）
            (SELECT address FROM user_info WHERE user_id = user_id LIMIT 1) -- 复用用户地址
        );
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;
CALL generate_order_main();
DROP PROCEDURE generate_order_main;

-- 5. 生成order_item（400万条，每个订单1-3个订单项）
DELIMITER //
CREATE PROCEDURE generate_order_item()
BEGIN
    DECLARE i INT DEFAULT 1;
    DECLARE goods_id INT;
    DECLARE goods_price DECIMAL(10,2);
    WHILE i <= 2000000 DO -- 对应order_main的200万订单
        -- 每个订单生成1-3个订单项
        SET @item_num = FLOOR(1+RAND()*3);
        WHILE @item_num > 0 DO
            SET goods_id = FLOOR(1+RAND()*100000); -- 关联goods的goods_id
            SET goods_price = (SELECT price FROM goods WHERE goods_id = goods_id LIMIT 1);
            INSERT INTO order_item (order_id, goods_id, goods_num, goods_price)
            VALUES (
                i, -- 关联order_main的order_id
                goods_id,
                FLOOR(1+RAND()*5), -- 商品数量1-5
                goods_price
            );
            SET @item_num = @item_num - 1;
        END WHILE;
        SET i = i + 1;
    END WHILE;
END //
DELIMITER ;
CALL generate_order_item();
DROP PROCEDURE generate_order_item;

