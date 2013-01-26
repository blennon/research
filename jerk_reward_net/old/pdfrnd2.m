%%pdfrnd(x, px, sampleSize): return a random sample of size sampleSize from 
%%the pdf px defined on the domain x. 

function [X] = pdfrnd2(x, px, sampleSize)

    cdf = cumsum(px);
    if abs(cdf(end) - 1) > 1e-5
        error('PDFrnd:ArgCheck','px not a pdf')
    end
    
    inds = zeros(1,sampleSize);
    for i=1:sampleSize
        z = rand;
%         if z < cdf(1)
%             inds(1,i) = 1;
%         else
%             inds(1,i) = find(cdf<z,1,'last');
%         end
        inds(1,i) = find(cdf>z,1,'first');
%         X = interp1(cdf, x, z, 'linear', 0);
%         if ceil(X) ~= inds(1,i)
%             X
%             inds(1,i)
%         end
    end
    X = x(inds);   
end

