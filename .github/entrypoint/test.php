<?php
    $url = '${ACCESS_URL}';
    $key = '${ACCESS_API}';
    $secretKey = '{ACCESS_KEY}';
    
	$data = [
        'method' => 'trade',
        'timestamp' => '1578304294000',
        'recvWindow' => '1578303937000',
        'pair' => 'btc_idr',
        'type' => 'sell',
        'price' => '1700000000',
        'idr' => '',
        'btc' => '0.00000863'
    ];
	$post_data = http_build_query($data, '', '&');
    $sign = hash_hmac('sha512', $post_data, $secretKey);

    $headers = ['Key:'.$key,'Sign:'.$sign];

    $curl = curl_init();

    curl_setopt_array($curl, array(
        CURLOPT_HTTPHEADER => $headers,
        CURLOPT_URL => $url,
        CURLOPT_POST => true,
        CURLOPT_POSTFIELDS => $data,
        CURLOPT_RETURNTRANSFER => true
    ));

    $response = curl_exec($curl);

    curl_close($curl);
    echo $response;
