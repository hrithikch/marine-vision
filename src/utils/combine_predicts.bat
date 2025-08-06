mkdir runs\sample_predict\all
for /R runs\sample_predict %f in (*.jpg) do @move "%f" runs\sample_predict\all\
