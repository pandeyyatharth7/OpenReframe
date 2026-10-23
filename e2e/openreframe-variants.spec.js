
const { test, expect } = require('@playwright/test');
const path = require('path');

test('OpenReframe generates, selects, and downloads a variant', async ({ page }) => {
  const imagePath = process.env.OPENREFRAME_TEST_IMAGE;
  expect(imagePath, 'Set OPENREFRAME_TEST_IMAGE first').toBeTruthy();

  await page.goto('http://127.0.0.1:8000/');
  await page.locator('#file').setInputFiles(path.resolve(imagePath));

  // Wait for the source image to load.
  await expect.poll(() =>
    page.locator('#iOrig').evaluate(img => img.naturalWidth)
  ).toBeGreaterThan(0);

  // Enter Smart mode and wait for planning to finish.
  await page.locator('#mSmart').click();
  await expect(page.locator('#candList .cand').first())
    .toBeVisible({ timeout: 180000 });

  // Generate just one variant to reduce processing time.
  await page.locator('#nVar').selectOption('1');

  // Keep one candidate selected and deselect the rest.
  const candidates = page.locator('#candList .cand input[type="checkbox"]');
  const count = await candidates.count();

  for (let i = 1; i < count; i++) {
    await candidates.nth(i).uncheck();
  }

  await page.locator('#genVars').click();

  // Wait for a variant card to appear.
  const card = page.locator('#grid .card').first();
  await expect(card).toBeVisible({ timeout: 60000 });

  // Wait for actual completion, not merely a queued card.
  await expect(card.locator('.title')).toContainText('done', {
    timeout: 600000,
  });

  // Select the completed variant.
  await card.getByRole('button', { name: 'Select' }).click();
  await expect(page.locator('#finalWrap')).toBeVisible();

  // Verify the selected output image loaded.
  await expect.poll(() =>
    page.locator('#iFinal').evaluate(img => img.naturalWidth)
  ).toBeGreaterThan(0);

  // Verify the final download link has a usable destination.
  const downloadLink = page.locator('#dlFinal');
  await expect(downloadLink).toHaveAttribute('href', /download=1/);

  // Verify the browser can actually download the image.
  const downloadPromise = page.waitForEvent('download');
  await downloadLink.click();
  const download = await downloadPromise;

  expect(download.suggestedFilename()).toBeTruthy();

  console.log('PASS: variant generated, selected, and downloaded.');
});
